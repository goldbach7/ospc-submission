# warm start to download model weights

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_gpu=False, max_batch_size=1, use_angle_cls=True, lang='ch')