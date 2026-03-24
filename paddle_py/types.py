from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TypedDict


class Box(TypedDict):
    x: int
    y: int
    width: int
    height: int


class RecognitionItem(TypedDict):
    text: str
    box: Box
    confidence: float


@dataclass(frozen=True)
class ModelSource:
    local_path: str
    url: str | None = None


LinesResult = List[List[RecognitionItem]]
