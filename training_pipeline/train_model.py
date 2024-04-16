from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import os

max_seq_length = 2048
MODEL_DIR = os.getenv('MODEL_DIR')

# change this dataset name to the one that is generated
dataset = load_from_disk('dataset_10Apr2024/')
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

promptFormat = """Check if there is unsafe content in the following message. Unsafe content is defined in the following categories.
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

Provide your safety assessment for the message above. Your first line MUST read 'safe' or 'unsafe'."""

# Formatting dataset to mimic output from OCR
def createText(example):
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant deciding whether a piece of content is safe or unsafe based on specific criteria. The content may contain irrelevant words as it has been extracted from an image."})
    messages.append(
        {"role": "user", "content": promptFormat.format(example['text'].strip())}
    )
    if example['label'] == 'offensive':
        messages.append({'role': 'assistant', 'content':'unsafe'})
    elif example['label'] == 'ok':
        messages.append({'role': 'assistant', 'content':'safe'})
    else:
        raise Exception('failed')
    prompt = tokenizer.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=False
    )
    example['full_instruction'] = prompt
    return example

# Formatting dataset to mimic OCR + caption format
def createFull(example):
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant deciding whether a piece of content is safe or unsafe based on specific criteria. The content may contain irrelevant words as it has been extracted from an image."})
    subPrompt = 'Image: {}, Caption: {}'.format(
        example['image'],
        example['text']
    )
    messages.append(
        {"role": "user", "content": promptFormat.format(subPrompt.strip())}
    )
    if example['label'] == 'offensive':
        messages.append({'role': 'assistant', 'content':'unsafe'})
    elif example['label'] == 'ok':
        messages.append({'role': 'assistant', 'content':'safe'})
    else:
        raise Exception('failed')
    prompt = tokenizer.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=False
    )
    example['full_instruction'] = prompt
    return example

# combine both datasets
image_dataset = dataset.map(createFull)
text_dataset = dataset.map(createText)
full_dataset = concatenate_datasets([image_dataset,text_dataset])
full_dataset = full_dataset.train_test_split(test_size=0.05)

# load model and tokenizer using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_DIR,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# load peft settings
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# define trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = full_dataset['train'],
    eval_dataset = full_dataset['test'],
    dataset_text_field = "full_instruction",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        eval_steps = 250,
        per_device_train_batch_size = 24,
        per_device_eval_batch_size= 24,
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        max_steps = 1600,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        save_steps = 250,
        output_dir = "outputs_openchat_10apr",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)
# start model training
trainer.train()

# change model name to your desired model name
model.save_pretrained_merged("trained_model", tokenizer, save_method = "merged_16bit",)