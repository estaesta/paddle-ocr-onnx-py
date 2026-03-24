"""Backward-compatible module path.

Deprecated: import from `paddle_py.ocr_engine` instead.
"""

from .ocr_engine import PaddleOcrServer, PaddleOnnxOCR

__all__ = ["PaddleOcrServer", "PaddleOnnxOCR"]
