from paddle_py import PaddleOnnxOCR

DET = "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/detection/PP-OCRv5_mobile_det_infer.onnx"
REC = "https://media.githubusercontent.com/media/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/recognition/PP-OCRv5_mobile_rec_infer.onnx"
DICT = "https://raw.githubusercontent.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr-models/main/recognition/ppocrv5_dict.txt"

ocr = PaddleOnnxOCR(det_model_path=DET, rec_model_path=REC, dict_path=DICT, debug=False)
ocr.initialize()

# Replace with your image path
result = ocr.recognize("/tmp/pi-clipboard-a76fd788-7952-48ff-898b-1575339d74ed.png")
print(result["text"])

ocr.destroy()
