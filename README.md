# paddle-ocr-onnx-py

Lightweight, in-process Python OCR engine using ONNX Runtime.

This project ports the practical OCR pipeline used in the TypeScript ecosystem to Python:
- text detection from the PP-OCR probability map
- contour-based text boxes with padding
- recognition using PP-OCRv5 recognition model
- CTC greedy decoding

Designed for direct in-process OCR integration.

## Install

### From local source

```bash
pip install -e .
```

### From GitHub

```bash
pip install "git+https://github.com/estaesta/paddle-ocr-onnx-py.git@v0.1.0"
```

## Quick usage

```python
from paddle_py import PaddleOnnxOCR

det = "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/detection/PP-OCRv5_mobile_det_infer.onnx"
rec = "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/recognition/PP-OCRv5_mobile_rec_infer.onnx"
dict_url = "https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/recognition/ppocrv5_dict.txt"

ocr = PaddleOnnxOCR(det_model_path=det, rec_model_path=rec, dict_path=dict_url)
ocr.initialize()
result = ocr.recognize("image.png")
print(result["text"])
ocr.destroy()
```

## Model caching

If a model path is a URL, the file is downloaded only when missing to:

`~/.cache/ppu-paddle-ocr`

## Credits

This library is inspired by and derived from ideas and pipeline behavior in:

- https://github.com/PT-Perkasa-Pilar-Utama/ppu-ocv
- https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr

Please check and follow licenses from those upstream projects when redistributing.
