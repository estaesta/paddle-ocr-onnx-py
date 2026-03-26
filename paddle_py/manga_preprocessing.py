"""
Manga-specific image preprocessing for better OCR on stylized text and complex backgrounds.
"""

import cv2
import numpy as np
from typing import Optional


def enhance_manga_text(
    img: np.ndarray,
    contrast: float = 1.8,
    brightness: int = 30,
    denoise: bool = True,
    clahe: bool = True,
) -> np.ndarray:
    """
    Enhance manga page for better text detection and recognition.
    
    Args:
        img: Input image (BGR or RGB)
        contrast: Contrast multiplier (1.0 = no change, >1 = more contrast)
        brightness: Brightness offset (-100 to 100)
        denoise: Apply denoising for complex backgrounds
        clahe: Apply CLAHE for better local contrast
    
    Returns:
        Enhanced image
    """
    result = img.copy()
    
    # Step 1: Increase contrast (helps with handwritten/stylized text)
    if contrast != 1.0 or brightness != 0:
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)
    
    # Step 2: Denoise for complex backgrounds (text over artwork)
    if denoise:
        result = cv2.fastNlMeansDenoisingColored(
            result, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )
    
    # Step 3: CLAHE for local contrast enhancement (better for varied lighting)
    if clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe_obj.apply(l)
        result = cv2.merge([l, a, b])
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    return result


def binarize_for_detection(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Convert to binary for better text detection on complex backgrounds.
    
    Args:
        img: Input image
        method: "otsu", "adaptive", or "sauvola"
    
    Returns:
        Binary image (0 and 255)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "sauvola":
        # Sauvola is better for text on textured backgrounds
        window_size = 25
        k = 0.2
        mean = cv2.boxFilter(gray, cv2.CV_32F, (window_size, window_size))
        sqr_mean = cv2.boxFilter(gray ** 2, cv2.CV_32F, (window_size, window_size))
        std = np.sqrt(sqr_mean - mean ** 2)
        threshold = mean * (1 + k * ((std / 128) - 1))
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    return binary


def remove_screentone(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Remove manga screentone patterns that interfere with text detection.
    
    Screentones are halftone patterns used for shading in manga.
    
    Args:
        img: Input image
        kernel_size: Morphological kernel size
    
    Returns:
        Image with reduced screentone
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use morphological opening to remove small dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # Convert back to BGR
    result = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    return result


def sharpen_text(img: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Sharpen text edges for better recognition of handwritten/stylized fonts.
    
    Args:
        img: Input image
        strength: Sharpening strength (1.0 = moderate, 2.0 = strong)
    
    Returns:
        Sharpened image
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]) * strength
    
    # Normalize to prevent overflow
    kernel = kernel / kernel.sum() * 5
    
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def preprocess_manga_pipeline(
    img: np.ndarray,
    profile: str = "default",
) -> np.ndarray:
    """
    Full preprocessing pipeline for manga pages.
    
    Args:
        img: Input image (BGR or RGB)
        profile: Preset profile ("default", "handwritten", "complex_bg", "screentone")
    
    Returns:
        Preprocessed image ready for OCR
    """
    if profile == "default":
        # General manga preprocessing
        result = enhance_manga_text(img, contrast=1.8, brightness=30, denoise=True, clahe=True)
    
    elif profile == "handwritten":
        # For handwritten/stylized fonts
        result = enhance_manga_text(img, contrast=2.2, brightness=40, denoise=False, clahe=True)
        result = sharpen_text(result, strength=1.8)
    
    elif profile == "complex_bg":
        # For text over complex artwork (not in bubbles)
        result = enhance_manga_text(img, contrast=2.0, brightness=50, denoise=True, clahe=True)
        result = sharpen_text(result, strength=1.3)
    
    elif profile == "screentone":
        # For pages with heavy screentone patterns
        result = remove_screentone(img, kernel_size=3)
        result = enhance_manga_text(result, contrast=1.6, brightness=20, denoise=False, clahe=True)
    
    else:
        raise ValueError(f"Unknown profile: {profile}")
    
    return result
