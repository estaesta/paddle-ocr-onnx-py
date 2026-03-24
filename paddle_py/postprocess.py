from __future__ import annotations

from typing import List, Tuple
import re

from .types import LinesResult, RecognitionItem


def normalize_common_ocr_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\boCR\b", "OCR", text)
    text = re.sub(r"\bGsk-WARNING\b", "GSK-WARNING", text)
    text = re.sub(r"\bgL\b", "GL", text)
    text = re.sub(r"\bGSk_RENDERER\b", "GSK_RENDERER", text)
    return text


def is_likely_single_char_noise(text: str, confidence: float) -> bool:
    token = text.strip()
    if len(token) != 1:
        return False
    if not token.isascii() or not token.isalpha():
        return False
    return confidence < 0.6


def sort_by_reading_order(results: List[RecognitionItem]) -> List[RecognitionItem]:
    return sorted(results, key=lambda r: (r["box"]["y"], r["box"]["x"]))


def group_results_by_line(results: List[RecognitionItem]) -> Tuple[LinesResult, str, float]:
    if not results:
        return [], "", 0.0

    lines: LinesResult = []
    current = [results[0]]
    avg_h = results[0]["box"]["height"]
    full_text = results[0]["text"]

    for i in range(1, len(results)):
        now = results[i]
        prev = results[i - 1]
        gap = abs(now["box"]["y"] - prev["box"]["y"])

        if gap <= avg_h * 0.5:
            current.append(now)
            full_text += f" {now['text']}"
            avg_h = sum(r["box"]["height"] for r in current) / len(current)
        else:
            lines.append(sorted(current, key=lambda r: r["box"]["x"]))
            current = [now]
            full_text += f"\n{now['text']}"
            avg_h = now["box"]["height"]

    if current:
        lines.append(sorted(current, key=lambda r: r["box"]["x"]))

    total_items = sum(len(line) for line in lines)
    total_conf = sum(sum(item["confidence"] for item in line) for line in lines)
    avg_conf = total_conf / total_items if total_items else 0.0
    return lines, full_text, avg_conf
