from __future__ import annotations

from typing import List, Tuple

import numpy as np
import cv2

from .types import Box


def calculate_resize(width: int, height: int, max_side_len: int) -> tuple[int, int, float]:
    ratio = 1.0
    resize_w, resize_h = width, height
    if max(width, height) > max_side_len:
        ratio = max_side_len / float(max(width, height))
        resize_w = int(round(width * ratio))
        resize_h = int(round(height * ratio))
    return resize_w, resize_h, ratio


def prepare_detection_canvas(img: np.ndarray, max_side_len: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    resize_w, resize_h, ratio = calculate_resize(w, h, max_side_len)
    resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

    target_w = int(np.ceil(resize_w / 32) * 32)
    target_h = int(np.ceil(resize_h / 32) * 32)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:resize_h, :resize_w] = resized
    return canvas, ratio, target_w, target_h


def image_to_tensor(canvas: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    normalized = canvas.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    chw = normalized.transpose(2, 0, 1).astype(np.float32)
    return np.expand_dims(chw, axis=0)


def normalize_detection_output(output: np.ndarray) -> np.ndarray:
    arr = np.array(output)
    if arr.ndim == 4 and arr.shape[1] == 1:
        prob = arr[0, 0]
    elif arr.ndim == 4 and arr.shape[3] == 1:
        prob = arr[0, :, :, 0]
    elif arr.ndim == 3:
        prob = arr[0]
    else:
        prob = arr.squeeze()
    return np.clip(prob, 0.0, 1.0)


def boxes_from_probability_map(
    prob_map: np.ndarray,
    ratio: float,
    max_w: int,
    max_h: int,
    orig_w: int,
    orig_h: int,
    min_box_area: int,
    padding_v: float,
    padding_h: float,
) -> List[Box]:
    gray = (prob_map * 255).astype(np.uint8)
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h <= min_box_area:
            continue

        vpad = int(round(h * padding_v))
        hpad = int(round(h * padding_h))

        px = max(0, x - hpad)
        py = max(0, y - vpad)
        right = min(max_w, x + w + hpad)
        bottom = min(max_h, y + h + vpad)

        pw = right - px
        ph = bottom - py

        ox = max(0, int(round(px / ratio)))
        oy = max(0, int(round(py / ratio)))
        ow = min(orig_w - ox, int(round(pw / ratio)))
        oh = min(orig_h - oy, int(round(ph / ratio)))

        if ow > 5 and oh > 5:
            boxes.append(Box(x=ox, y=oy, width=ow, height=oh))

    return boxes


def save_detection_debug(debug_dir: str, src_img: np.ndarray, prob_map: np.ndarray, boxes: List[Box]) -> None:
    import os

    os.makedirs(debug_dir, exist_ok=True)
    gray = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(debug_dir, "detection-debug.png"), gray)

    canvas = cv2.cvtColor(src_img.copy(), cv2.COLOR_RGB2BGR)
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(debug_dir, "boxes-debug.png"), canvas)
