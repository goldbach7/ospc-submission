FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get -y --no-install-recommends install git build-essential

## Copy weights here
COPY openchat-model-ospc/ /opt/models/llm/
COPY blip-image-captioning-base/ /opt/models/blip/

## install and warmup surya
RUN python -m pip install pydantic git+https://github.com/billcai/surya.git@f486f4551dbb5ad99db440344f152718aadd912b
RUN python -m pip install opencv-python-headless

COPY warm_start.py .
RUN python warm_start.py

## install edited version with no logging
RUN python -m pip uninstall --yes surya-ocr && python -m pip install git+https://github.com/billcai/surya.git@a9b0fa6659de1a55b3f80fd0be94338ea16be89d
# install PaddleOCR
RUN python -m pip install "paddlepaddle==2.5.2" -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install "paddleocr==2.7.0.3"
RUN python -m pip uninstall --yes opencv-python
RUN python -m pip install "opencv-python-headless<=4.6.0.66"

COPY warm_start_paddleocr.py .
RUN python warm_start_paddleocr.py

# install transformers and other packages
RUN python -m pip install -U transformers
RUN python -m pip install optimum accelerate auto-gptq
RUN python -m pip install gpustat

COPY evaluate-gptq-openchat-use-logits-trained.py .
COPY prompt_format_utils.py .
COPY surya_ocr.py .
COPY paddle_ocr.py .
ENTRYPOINT ["python", "evaluate-gptq-openchat-use-logits-trained.py"]
