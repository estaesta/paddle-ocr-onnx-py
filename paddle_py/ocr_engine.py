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
