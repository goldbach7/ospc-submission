from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from PIL import Image
from surya.ocr import run_ocr
import logging

logging.basicConfig(level=logging.INFO)

det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()

# Run OCR give image path
def ocr_image(image_path):
    try:
        image = Image.open(image_path)
        langs = ['en', 'zh', 'ta', 'ms']

        predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
        lines = ' '.join([x.text for x in predictions[0].text_lines])
        return lines
    except Exception as e:
        logging.info('failed for ocr')
        return None