# warm start to download model weights

from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

if __name__ == "__main__":
  det_processor, det_model = load_det_processor(), load_det_model()
  rec_model, rec_processor = load_rec_model(), load_rec_processor()