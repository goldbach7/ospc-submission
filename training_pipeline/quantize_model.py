from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
import torch
import os 

MODEL_DIR = os.getenv('MODEL_DIR')

# load 16bit model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)

# quantize model - this takes a while, around 30 minutes on RTX 3090
quantizer = GPTQQuantizer(bits=4, dataset='c4-new', block_name_to_quantize = "model.layers", model_seqlen = 4096)
quantized_model = quantizer.quantize_model(model, tokenizer)

# save 4bit model and tokenizer
quantizer.save(model,'quantized_model')
tokenizer.save_pretrained('quantized_model')