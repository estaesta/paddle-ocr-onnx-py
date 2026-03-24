from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def crop_to_tensor(crop: np.ndarray, rec_image_height: int) -> tuple[np.ndarray, tuple[int, int]]:
    if crop.ndim == 3 and crop.shape[2] >= 1:
        red = crop[:, :, 0]
    else:
        red = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    h, w = red.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((1, 3, rec_image_height, 8), dtype=np.float32), (h, w)

    target_h = rec_image_height
    target_w = max(8, int(round(target_h * (w / h))))
    resized = cv2.resize(red, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    chw = np.stack([norm, norm, norm], axis=0).astype(np.float32)
    return np.expand_dims(chw, axis=0), (h, w)


def to_2d_logits(output: np.ndarray) -> np.ndarray:
    arr = np.array(output)
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    return arr.squeeze()


def ctc_greedy_decode(output: np.ndarray, char_dict: List[str]) -> Tuple[str, float]:
    logits = to_2d_logits(output)
    seq_len, num_classes = logits.shape

    dictionary = list(char_dict)
    if num_classes > len(dictionary):
        dictionary.extend([f"<PAD_{i}>" for i in range(num_classes - len(dictionary))])

    decoded: List[str] = []
    confidences: List[float] = []
    last_idx = -1

    for t in range(seq_len):
        probs = logits[t]
        idx = int(np.argmax(probs))
        max_prob = float(np.max(probs))

        if idx == 0 or idx == last_idx:
            last_idx = idx
            continue
        if idx < 0 or idx >= len(dictionary):
            last_idx = idx
            continue

        token = dictionary[idx]
        if idx == len(dictionary) - 1:
            if token != "<unk>":
                decoded.append(" ")
                confidences.append(max_prob)
        else:
            decoded.append(token)
            confidences.append(max_prob)

        last_idx = idx

    text = "".join(decoded)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, confidence
