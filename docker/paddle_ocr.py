from paddleocr import PaddleOCR
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

ocr = PaddleOCR(use_gpu=False, max_batch_size=1, use_angle_cls=True, lang='ch',show_log = False) 

# Run PaddleOCR given image_path
def ocr_image(image_path):
    try:
        result = ocr.ocr(image_path, cls=True)
        logging.info(result)
        lines = ' '.join([''.join(x[1][0]) for x in result[0] if x[1][1] > 0.9])
        return lines
    except Exception as e:
        logging.info('failed for ocr: {}'.format(str(e)))
        return None