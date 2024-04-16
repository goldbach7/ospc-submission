
import base64, os, sys, json
from prompt_format_utils import SafetyCategory
from pydantic import BaseModel
import logging
import fileinput
from surya_ocr import ocr_image
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from paddle_ocr import ocr_image as paddle_ocr_image
import gpustat
from scipy.special import softmax

METHOD = os.getenv('INFERENCE_METHOD','logit_score')

## Logging is disabled and minimized during submmission
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.disabled = True

# Pydantic class to store visual description and OCR text
class VisualInformation(BaseModel):
    visual_description: str
    text_image: str

## Function to extract information from image caltion model
def get_caption(
    image_location,
    model
):
    image = Image.open(image_location)
    inputs = model['processor'](image, return_tensors="pt").to("cuda:0", torch.float16)
    out = model['model'].generate(**inputs)
    caption = model['processor'].decode(out[0], skip_special_tokens=True)
    return caption

## parse output for sampled decoding strategy
def parse_resp(
    rawText:str
):
    rawText = rawText.lower().split('\n')[0]
    if 'unsafe' in rawText:
        return 'unsafe'
    elif 'safe' in rawText:
        return 'safe'
    else:
        return None

## few shot prompts used for sampled decoding strategy
def example_prompt():
    promptTemplate = """Check if there is unsafe content in the following message. Unsafe content is defined in the following categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
Contains hate speech and harmful content. Harmful content centres on divisive and contentious subject matters. Typical areas include:
- Racial disparities
- Religious beliefs and practices
- Sexual orientation
- Nationalistic sentiments and xenophobia against migrants
- Socio-economic divides and class struggles
- Age-related biases
- Gender discrimination
- Discrimination based on illnesses and disabilities
These content may be based on Singapore context, including the use of non-English languages like Mandarin, Malay and Tamil.
<END UNSAFE CONTENT CATEGORIES>

This is the message you are checking if it contains unsafe content.
<BEGIN MESSAGE>
promptText
<END MESSAGE>

Provide your safety assessment for the message above. Your first line MUST read 'safe' or 'unsafe'.
""".format(
        "\n".join([f"{i+1}: {c.name}\n{c.description}" for i, c in enumerate(safetyCategory)])
    )
    safePrompt1 = promptTemplate.replace('promptText','Image: a person happy, Caption: Hasdasdello World, 我叫 goldbach imgflip.com')
    unsafePrompt1 = promptTemplate.replace('promptText','Image: a person angry, Caption: Selamat datang, aisuhdaiusd, semua orang cina jahat imgflip.com')
    safePrompt2 = promptTemplate.replace('promptText','Image: a monkey with a hat with text asiduhnasasd, Caption: இது ஒரு வேடிக்கையான படம் 恨坏')
    unsafePrompt2 = promptTemplate.replace('promptText','Image: a child laughing with text 很好, Caption: 恨死资本家')
    
    return [
        {'role': 'user', 'content': safePrompt1},
        {'role': 'assistant', 'content': 'safe'},
        {'role': 'user', 'content': unsafePrompt1},
        {'role': 'assistant', 'content': 'unsafe'},
        {'role': 'user', 'content': unsafePrompt2},
        {'role': 'assistant', 'content': 'unsafe'},
        {'role': 'user', 'content': safePrompt2},
        {'role': 'assistant', 'content': 'safe'},
    ]

safetyCategory = [
            SafetyCategory(
                name='Offensive Content',
                description="""Contains hate speech and harmful content. Harmful content centres on divisive and contentious subject matters. Typical areas include:
- Racial disparities
- Religious beliefs and practices
- Sexual orientation
- Nationalistic sentiments and xenophobia against migrants
- Socio-economic divides and class struggles
- Age-related biases
- Gender discrimination
- Discrimination based on illnesses and disabilities
These content may be based on Singapore context, including the use of non-English languages like Mandarin, Malay and Tamil."""
            )
        ]

## Query LLM to extract harmfulness classification
def query_llm(
    promptText,
    safetyCategory,
    llm,
    tokenizer
):
    ## this is used for logit method
    if METHOD=='logit_score':
        prompt = """Check if there is unsafe content in the following message. Unsafe content is defined in the following categories.
    <BEGIN UNSAFE CONTENT CATEGORIES>
    Contains hate speech and harmful content. Harmful content centres on divisive and contentious subject matters. Typical areas include:
    - Racial disparities
    - Religious beliefs and practices
    - Sexual orientation
    - Nationalistic sentiments and xenophobia against migrants
    - Socio-economic divides and class struggles
    - Age-related biases
    - Gender discrimination
    - Discrimination based on illnesses and disabilities
    These content may be based on Singapore context, including the use of non-English languages like Mandarin, Malay and Tamil.
    <END UNSAFE CONTENT CATEGORIES>

    This is the message you are checking if it contains unsafe content.
    <BEGIN MESSAGE>
    {}
    <END MESSAGE>

    Provide your safety assessment for the message above. Your first line MUST read 'safe' or 'unsafe'.
    """.format(
            promptText
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant deciding whether a piece of content is safe or unsafe based on specific criteria. The content may contain irrelevant words as it has been extracted from an image."},
        ] + [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
        generation_output = llm.generate(
            model_inputs.input_ids,
            max_new_tokens=10,
            temperature=0.1,
            output_logits=True,
            return_dict_in_generate=True
        )
        generated_sequences = generation_output['sequences']
        generated_logits = generation_output['logits']
        unsafeTokenId = tokenizer.encode('unsafe')[1]
        safeTokenId = tokenizer.encode('safe')[1]
        firstLogit = generated_logits[0].cpu().numpy()
        prob = softmax([
            firstLogit[0,unsafeTokenId],
            firstLogit[0,safeTokenId],
            ])
        return prob
    else:
        ## this is used for sampled decoding
        prompt = """Check if there is unsafe content in the following message. Unsafe content is defined in the following categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
{}
<END UNSAFE CONTENT CATEGORIES>

This is the message you are checking if it contains unsafe content.
<BEGIN MESSAGE>
{}
<END MESSAGE>

Provide your safety assessment for the message above. Your first line MUST read 'safe' or 'unsafe'.
""".format(
        "\n".join([f"{i+1}: {c.name}\n{c.description}" for i, c in enumerate(safetyCategory)]),
        promptText
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant deciding whether a piece of content is safe or unsafe based on specific criteria. The content may contain irrelevant words as it has been extracted from an image."},
    ] + example_prompt() + [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = llm.generate(
        model_inputs.input_ids,
        max_new_tokens=20,
        temperature=0.1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsedResp = parse_resp(resp)
    return parsedResp


def process_llm(
    visualInfo: VisualInformation,
    query_function,
    llm,
    tokenizer
):
    promptText = visualInfo.text_image
    textResult = query_llm(promptText,safetyCategory,llm,tokenizer)
    
    promptText = "Image: {}, Caption: {}".format(visualInfo.visual_description, visualInfo.text_image)
    visualTextResult = query_llm(promptText,safetyCategory,llm,tokenizer)

    average_result = (textResult[0] + visualTextResult[0]) / 2
    return average_result

def run_for_single_image(
    img_location,
    caption_model,
    llm,
    tokenizer,
    DEFAULT = 0.5# logger.disabled = True
):
    try:
        resp = get_caption(img_location,caption_model)
    except Exception as e:
        logger.disabled = False
        logger.warn('captioning failed: error is {}'.format(e))
        logger.disabled = True
        resp = None 
    ocrResult = ''
    try:
        ocrResult += ocr_image(img_location)
    except Exception as e:
        logger.disabled = False
        logger.warn('surya ocr failed: error is {}'.format(e))
        logger.disabled = True
    try:
        ocrResult += ' ' + paddle_ocr_image(img_location)
    except Exception as e:
        logger.disabled = False
        logger.warn('paddle ocr failed: error is {}'.format(e))
        logger.disabled = True
    if len(ocrResult) == 0:
        return DEFAULT
    if resp is None:
        resp = ""
    resp = VisualInformation(visual_description=resp,text_image=ocrResult)
    logger.info(resp)
    try:
        llmResult = process_llm(resp, query_llm, llm, tokenizer)
        logger.info('LLM result for {} is {}'.format(img_location,llmResult))
    except Exception as e:
        logger.disabled = False
        logger.warn('hit error with llm result for {}: {}'.format(img_location,str(e)))
        logger.disabled = True 
        return DEFAULT
    return llmResult

if __name__ == "__main__":
    processor = BlipProcessor.from_pretrained("/opt/models/blip/")
    blipModel = BlipForConditionalGeneration.from_pretrained("/opt/models/blip/", torch_dtype=torch.float16).to("cuda:0")
    caption_model = {'model':blipModel, 'processor':processor}
    # logger.disabled = True
    llm = AutoModelForCausalLM.from_pretrained(
        "/opt/models/llm/",
        # "/home/tars/memes/memes2024/Mistral-7B-Instruct-v0.2-GPTQ/",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/opt/models/llm/"
        # "/home/tars/memes/memes2024/Mistral-7B-Instruct-v0.2-GPTQ/"
        )

    for line in fileinput.input(encoding="utf-8"):
        # IMPORTANT: Please ensure any trailing whitespace (eg: \n) is removed. This may impact some modules to open the filepath
        logger.info(line)
        image_path = line.rstrip()

        try:
            # Process the image
            start_time = time.time()
            res = run_for_single_image(
                image_path,
                caption_model,
                llm,
                tokenizer
                )
            label = int(res > 0.5)
            end_time = time.time()# logger.disabled = True
            logger.disabled = False
            logger.warn('time taken to process: {}'.format(end_time-start_time))
            try:
                logger.warn(gpustat.GPUStatCollection.new_query(debug=False,id=None).jsonify())
            except Exception as e:
                logger.warn('gpu loggin failed: {}'.format(e))
            logger.disabled = True

            # Ensure each result for each image_path is a new line
            sys.stdout.write(f"{res:.4f}\t{label}\n")

        except Exception as e:
            # Output to any raised/caught error/exceptions to stderr
            sys.stderr.write(str(e))