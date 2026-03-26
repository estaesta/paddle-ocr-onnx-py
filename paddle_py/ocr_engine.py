"""Production-ready ONNX Paddle OCR engine."""

from __future__ import annotations

from io import BytesIO
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .detection import (
    boxes_from_probability_map,
    image_to_tensor,
    normalize_detection_output,
    prepare_detection_canvas,
    save_detection_debug,
)
from .postprocess import (
    group_results_by_line,
    is_likely_single_char_noise,
    normalize_common_ocr_text,
    sort_by_reading_order,
)
from .recognition import crop_to_tensor, ctc_greedy_decode
from .resources import ensure_local_resource, resolve_source
from .types import Box, RecognitionItem

try:
    from .manga_preprocessing import preprocess_manga_pipeline
except ImportError:
    preprocess_manga_pipeline = None


class PaddleOnnxOCR:
    def __init__(
        self,
        det_model_path: Optional[str] = None,
        rec_model_path: Optional[str] = None,
        dict_path: Optional[str] = None,
        use_gpu: bool = False,
        max_side_len: int = 640,
        debug: bool = False,
        lang: str = "auto",
        use_beam_search: bool = False,
        beam_width: int = 5,
        enable_vertical: bool = True,
        manga_mode: bool = False,
        manga_profile: str = "default",
        multiscale_detection: bool = False,
    ) -> None:
        self.det_source = resolve_source(
            det_model_path,
            "det_model_path must be provided as a URL or local cache path (no fallback).",
        )
        self.rec_source = resolve_source(
            rec_model_path,
            "rec_model_path must be provided as a URL or local cache path (no fallback).",
        )
        self.dict_source = resolve_source(
            dict_path,
            "dict_path must be provided as a URL or local cache path (no fallback).",
        )

        self.use_gpu = use_gpu
        self.max_side_len = max_side_len
        self.debug = debug
        self.lang = lang
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        self.enable_vertical = enable_vertical
        self.manga_mode = manga_mode
        self.manga_profile = manga_profile
        self.multiscale_detection = multiscale_detection

        self.det_session: Optional[ort.InferenceSession] = None
        self.rec_session: Optional[ort.InferenceSession] = None
        self.char_dict: List[str] = []

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.min_box_area = 25
        self.padding_v = 0.4
        self.padding_h = 0.6
        self.rec_image_height = 48
        self.vertical_aspect_ratio = 1.5

        # Detection parameters (tunable for manga)
        self.det_db_thresh = 0.3
        self.det_db_box_thresh = 0.5
        self.det_db_unclip_ratio = 1.5
        self.det_limit_side_len = 960
        
        # Manga-specific settings
        if self.manga_mode:
            self._apply_manga_settings()

    def _apply_manga_settings(self) -> None:
        """Apply optimized settings for manga text detection."""
        # More aggressive detection for text on complex backgrounds
        self.det_db_thresh = 0.2
        self.det_db_box_thresh = 0.45
        self.det_db_unclip_ratio = 2.0
        self.det_limit_side_len = 1280
        
        # Better for furigana and small text
        self.min_box_area = 16
        self.padding_v = 0.3
        self.padding_h = 0.5

    def initialize(self) -> None:
        if ort is None:
            raise ImportError("onnxruntime is not installed. Please install onnxruntime.")
        if cv2 is None:
            raise ImportError("opencv-python is required. Please install opencv-python.")

        ensure_local_resource(self.det_source)
        ensure_local_resource(self.rec_source)
        ensure_local_resource(self.dict_source)

        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            providers = ort.get_available_providers()

        self.det_session = ort.InferenceSession(self.det_source.local_path, providers=providers)
        self.rec_session = ort.InferenceSession(self.rec_source.local_path, providers=providers)

        with open(self.dict_source.local_path, "r", encoding="utf-8") as f:
            self.char_dict = [line.rstrip("\n") for line in f.readlines()]

    def is_initialized(self) -> bool:
        return self.det_session is not None and self.rec_session is not None

    def destroy(self) -> None:
        self.det_session = None
        self.rec_session = None

    def recognize(self, image: Any, flatten: bool = False) -> Dict[str, Any]:
        if not self.is_initialized():
            raise RuntimeError("PaddleOnnxOCR is not initialized. Call initialize() first.")

        img = np.array(self._load_image(image))
        
        # Apply manga preprocessing if enabled
        if self.manga_mode and preprocess_manga_pipeline is not None:
            img = preprocess_manga_pipeline(img, profile=self.manga_profile)
        
        base = self._recognize_image(img, flatten)

        if not self.enable_vertical:
            return base

        # Try rotated passes for vertical text and keep best result.
        rotated_cw = self._recognize_image(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), flatten)
        rotated_ccw = self._recognize_image(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), flatten)

        best = self._select_best_result([base, rotated_cw, rotated_ccw], flatten)
        return best

    def deskew_image(self, image: Any) -> Image.Image:
        pil = self._load_image(image)
        img = np.array(pil)
        _, prob_map, _ = self._detect(img, return_prob_map=True)
        if prob_map is None:
            return pil

        gray = (prob_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pil

        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        angle = rect[-1]
        if angle < -45:
            angle += 90

        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)

    def _load_image(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(BytesIO(image)).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise TypeError("image must be file path, bytes, or PIL.Image.Image")

    def _recognize_image(self, img: np.ndarray, flatten: bool) -> Dict[str, Any]:
        if self.multiscale_detection:
            boxes = self._detect_multiscale(img)
        else:
            boxes = self._detect(img)
            
        if not boxes:
            key = "results" if flatten else "lines"
            return {"text": "", key: [], "confidence": 0.0}

        results = sort_by_reading_order(self._recognize_boxes(img, boxes))
        lines, text, confidence = group_results_by_line(results)

        if flatten:
            return {"text": text.replace("\n", " "), "results": results, "confidence": confidence}
        return {"text": text, "lines": lines, "confidence": confidence}

    def _select_best_result(self, candidates: List[Dict[str, Any]], flatten: bool) -> Dict[str, Any]:
        def key(res: Dict[str, Any]) -> tuple[float, int]:
            text = res.get("text", "") or ""
            return float(res.get("confidence", 0.0)), len(text.strip())

        return sorted(candidates, key=key, reverse=True)[0]

    def _detect(self, img: np.ndarray, return_prob_map: bool = False):
        orig_h, orig_w = img.shape[:2]
        canvas, ratio, target_w, target_h = prepare_detection_canvas(img, self.max_side_len)
        tensor = image_to_tensor(canvas, self.mean, self.std)

        input_name = self.det_session.get_inputs()[0].name
        raw_output = self.det_session.run(None, {input_name: tensor})[0]
        prob_map = normalize_detection_output(raw_output)

        boxes = boxes_from_probability_map(
            prob_map=prob_map,
            ratio=ratio,
            max_w=target_w,
            max_h=target_h,
            orig_w=orig_w,
            orig_h=orig_h,
            min_box_area=self.min_box_area,
            padding_v=self.padding_v,
            padding_h=self.padding_h,
        )

        if self.debug:
            debug_dir = os.path.join(os.getcwd(), "paddle_onnx_debug")
            save_detection_debug(debug_dir, img, prob_map, boxes)

        if return_prob_map:
            return boxes, prob_map, ratio
        return boxes

    def _detect_multiscale(self, img: np.ndarray) -> List[Box]:
        """
        Detect text at multiple scales for better detection of varying text sizes.
        Useful for manga with large dialogue + tiny furigana.
        """
        scales = [1.0, 1.5]  # Original + 1.5x upscale
        all_boxes = []
        
        for scale in scales:
            if scale != 1.0:
                h, w = img.shape[:2]
                scaled_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            else:
                scaled_img = img
            
            boxes = self._detect(scaled_img)
            
            # Scale boxes back to original coordinates
            if scale != 1.0:
                for box in boxes:
                    box['x'] = int(box['x'] / scale)
                    box['y'] = int(box['y'] / scale)
                    box['width'] = int(box['width'] / scale)
                    box['height'] = int(box['height'] / scale)
            
            all_boxes.extend(boxes)
        
        # Remove duplicate/overlapping boxes using Non-Maximum Suppression
        return self._nms_boxes(all_boxes, iou_threshold=0.5)

    def _nms_boxes(self, boxes: List[Box], iou_threshold: float = 0.5) -> List[Box]:
        """Non-Maximum Suppression to remove duplicate detections."""
        if not boxes:
            return []
        
        # Sort by area (larger boxes first)
        boxes = sorted(boxes, key=lambda b: b['width'] * b['height'], reverse=True)
        
        keep = []
        while boxes:
            current = boxes.pop(0)
            keep.append(current)
            
            # Remove boxes that overlap significantly with current
            boxes = [box for box in boxes if self._box_iou(current, box) < iou_threshold]
        
        return keep

    def _box_iou(self, box1: Box, box2: Box) -> float:
        """Calculate Intersection over Union for two boxes."""
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = box1['width'] * box1['height']
        box2_area = box2['width'] * box2['height']
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _recognize_crop(self, crop: np.ndarray, rec_input: str) -> tuple[str, float]:
        tensor, _ = crop_to_tensor(crop, self.rec_image_height)
        output = self.rec_session.run(None, {rec_input: tensor})[0]
        return ctc_greedy_decode(output, self.char_dict)

    def _recognize_crop_best(self, crop: np.ndarray, rec_input: str) -> tuple[str, float]:
        h, w = crop.shape[:2]
        candidates: List[tuple[str, float]] = [self._recognize_crop(crop, rec_input)]

        if w > 0 and h / float(w) >= self.vertical_aspect_ratio:
            cw = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            ccw = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            candidates.append(self._recognize_crop(cw, rec_input))
            candidates.append(self._recognize_crop(ccw, rec_input))

        best_text, best_conf = candidates[0]
        for text, conf in candidates[1:]:
            if conf > best_conf:
                best_text, best_conf = text, conf
            elif conf == best_conf and len(text) > len(best_text):
                best_text, best_conf = text, conf
        return best_text, best_conf

    def _recognize_boxes(self, img: np.ndarray, boxes: List[Box]) -> List[RecognitionItem]:
        results: List[RecognitionItem] = []
        debug_dir = None
        if self.debug:
            debug_dir = os.path.join(os.getcwd(), "paddle_onnx_debug")
            os.makedirs(debug_dir, exist_ok=True)

        rec_input = self.rec_session.get_inputs()[0].name

        for idx, box in enumerate(boxes):
            x, y, w, h = box["x"], box["y"], box["width"], box["height"]
            crop = img[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            text, confidence = self._recognize_crop_best(crop, rec_input)
            text = normalize_common_ocr_text(text)
            if is_likely_single_char_noise(text, confidence):
                continue

            if debug_dir is not None:
                cv2.imwrite(os.path.join(debug_dir, f"crop_{idx:03d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            results.append({"text": text, "box": box, "confidence": confidence})

        return results


PaddleOcrServer = PaddleOnnxOCR
