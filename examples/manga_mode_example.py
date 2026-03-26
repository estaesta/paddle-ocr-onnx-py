"""
Example: Using manga mode for better OCR on manga pages.

This example demonstrates:
- Manga preprocessing profiles
- Multi-scale detection
- Optimized parameters for complex backgrounds
"""

import sys
from pathlib import Path

# Add paddle-py to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from paddle_py import PaddleOnnxOCR

# Model URLs (same as before)
DET_URL = 'https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/detection/PP-OCRv5_mobile_det_infer.onnx'
REC_URL = 'https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/recognition/PP-OCRv5_mobile_rec_infer.onnx'
DICT_URL = 'https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/recognition/ppocrv5_dict.txt'


def test_manga_mode(image_path: str):
    """Test different manga preprocessing profiles."""
    
    profiles = ["default", "handwritten", "complex_bg", "screentone"]
    
    print("=" * 80)
    print("MANGA OCR COMPARISON")
    print("=" * 80)
    
    # Test without manga mode
    print("\n[1] Without Manga Mode")
    print("-" * 80)
    ocr_normal = PaddleOnnxOCR(
        det_model_path=DET_URL,
        rec_model_path=REC_URL,
        dict_path=DICT_URL,
        enable_vertical=True,
        manga_mode=False,
    )
    ocr_normal.initialize()
    result_normal = ocr_normal.recognize(image_path)
    print(f"Text:\n{result_normal['text']}\n")
    print(f"Confidence: {result_normal['confidence']:.4f}")
    
    # Test with manga mode + different profiles
    for profile in profiles:
        print(f"\n[2.{profiles.index(profile) + 1}] Manga Mode: {profile}")
        print("-" * 80)
        
        ocr_manga = PaddleOnnxOCR(
            det_model_path=DET_URL,
            rec_model_path=REC_URL,
            dict_path=DICT_URL,
            enable_vertical=True,
            manga_mode=True,
            manga_profile=profile,
            multiscale_detection=False,  # Can enable for better furigana detection
        )
        ocr_manga.initialize()
        result_manga = ocr_manga.recognize(image_path)
        print(f"Text:\n{result_manga['text']}\n")
        print(f"Confidence: {result_manga['confidence']:.4f}")
    
    # Test with multiscale detection
    print(f"\n[3] Manga Mode + Multiscale Detection")
    print("-" * 80)
    ocr_multiscale = PaddleOnnxOCR(
        det_model_path=DET_URL,
        rec_model_path=REC_URL,
        dict_path=DICT_URL,
        enable_vertical=True,
        manga_mode=True,
        manga_profile="default",
        multiscale_detection=True,
    )
    ocr_multiscale.initialize()
    result_multiscale = ocr_multiscale.recognize(image_path)
    print(f"Text:\n{result_multiscale['text']}\n")
    print(f"Confidence: {result_multiscale['confidence']:.4f}")


def standalone_preprocessing_example(image_path: str):
    """Example of using preprocessing independently."""
    import cv2
    from paddle_py.manga_preprocessing import (
        enhance_manga_text,
        binarize_for_detection,
        sharpen_text,
        preprocess_manga_pipeline,
    )
    
    print("\n" + "=" * 80)
    print("STANDALONE PREPROCESSING EXAMPLE")
    print("=" * 80)
    
    img = cv2.imread(image_path)
    
    # Apply different preprocessing steps
    enhanced = enhance_manga_text(img, contrast=1.8, brightness=30)
    sharpened = sharpen_text(enhanced, strength=1.5)
    
    # Or use the full pipeline
    preprocessed = preprocess_manga_pipeline(img, profile="handwritten")
    
    # Save results for inspection
    cv2.imwrite("preprocessed_enhanced.png", enhanced)
    cv2.imwrite("preprocessed_sharpened.png", sharpened)
    cv2.imwrite("preprocessed_pipeline.png", preprocessed)
    
    print("Saved preprocessed images:")
    print("  - preprocessed_enhanced.png")
    print("  - preprocessed_sharpened.png")
    print("  - preprocessed_pipeline.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python manga_mode_example.py <image_path>")
        print("\nExample:")
        print("  python manga_mode_example.py /path/to/manga_page.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Test manga mode with different profiles
    test_manga_mode(image_path)
    
    # Show standalone preprocessing
    standalone_preprocessing_example(image_path)
