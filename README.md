# OSPC Code and Models

Submission Details

- Author: Bill Cai
- Final AUROC: 0.7600

## Submission structure

The following structure explains the submitted files:
```
submission/
    docker/ # contains all files required to build submission container
        blip-image-captioning-base/ # cloned from https://huggingface.co/Salesforce/blip-image-captioning-base
        openchat-model-ospc/ # model weights of highest AUROC fine-tuned LLM quantized to 4bit with GPTQ, clone from https://huggingface.co/goldbach7/openchat-model-ospc
        gptq.dockerfile # dockerfile to build container
        evaluate-gptq-openchat-use-logits-trained.py # main inference file used to generate predictions
        ...
    data_preparation/ # contains all files required to generate synthetic dataset
        generate_data.py # Python script used to generate English synthetic data from Together AI
        translate_data.py # Python script used to translate data using models from Together AI
        translate_data_api.py # Python script used 
        ...
    training_pipeline/ # contains Python scripts used to fine-tune and quantize models
        train_model.py # Python script used for QLoRA fine-tuning
        load_lora.py # Python script used for loading and merge specific LoRA checkpoints
        quantize_model.py # Python script used for quantizing models, including fine-tuned and pretrained models
    datasets/ # datasets used, saved in HuggingFace datasets format
        v1/ # version 1 of dataset, 1216 examples
        v2/ # version 2 of dataset, 2734 examples
        v3/ # version 3 of dataset, 6318 examples
    README.md # this file
```

## Building Docker container

Make sure you have Git LFS installed. You should first clone BLIP and the fine-tuned LLMs with:
```
cd model_package
git clone https://huggingface.co/goldbach7/openchat-model-ospc
git clone https://huggingface.co/Salesforce/blip-image-captioning-base
```

You can build the Docker container with:
```
cd model_package
docker build . -t memes2024:gptq-final-submission
```

Using the [AISG submission guide](https://github.com/AISG-Technology-Team/AISG-Online-Safety-Challenge-Submission-Guide), you can test the solution. Do create the isolated network as stated in the guide.
```
cd 
git clone https://github.com/AISG-Technology-Team/AISG-Online-Safety-Challenge-Submission-Guide
cd AISG-Online-Safety-Challenge-Submission-Guide
cat local_test/test_stdin/stdin.csv | \
docker run --init \
        --attach "stdin" \
        --attach "stdout" \
        --attach "stderr" \
        --cpus 2 \
        --gpus "device=0" \
        --memory 4g \
        --memory-swap 0 \
        --ulimit nproc=1024 \
        --ulimit nofile=1024 \
        --network exec_env_jail_network \
        --read-only \
        --mount type=bind,source="$(pwd)"/local_test/test_images,target=/images,readonly \
        --mount type=tmpfs,destination=/tmp,tmpfs-size=5368709120,tmpfs-mode=1777 \
        --interactive \
        memes2024:gptq-final-submission \
 1>local_test/test_output/stdout.csv \
 2>local_test/test_output/stderr.csv
```

## Generating synthetic data

First, sign up for an account at Together AI that comes with free credits. You can then generate synthetic data using one of the prompts specified in `data_generation/promptDict.py`. For example,

```
MODEL_ID="mistralai/Mixtral-8x7B-Instruct-v0.1" API_KEY="<fill in your Together AI API key here>" PROMPT_KEY="general_ok" python generate_data.py
```

You can use the environment variable `NUM_RUNS` to increase the number of synthetic data. By default, the LLM is prompted to generate 5 pieces of synthetic data per invocation, and the LLM is by default invoked 10 times. Each file will be stored in a folder that will be automatically created.

Then, you can translate the dataset generated using either a Together AI model or Google Cloud API. Google Cloud API allows for free-tier usage of its translation API, but requires an API key. You will need to provide the folder name from the previous step as input.

**For Together AI translation**:
```
MODEL_ID="mistralai/Mixtral-8x7B-Instruct-v0.1" API_KEY="<fill in your Together AI API key here>" LANGUAGE="<type in the full spelling of the language, e.g. Tamil>" DIR_NAME="<directory name of folder generated in previous step>" python3 translate_data.py
```

**For Google Cloud translation**:
```
API_KEY="<fill in your Google Cloud API key>" LANGUAGE="<choose between zh (Chinese), ms (Malay) or ta (Tamil)>" DIR_NAME="<directory name of folder generated in previous step>" python3 translate_data_api.py
```

Next, after calling the synthetic generation script your desired number of times through the different prompt types, you can call `python3 clean_dataset.py` which will automatically go through all the generated dataset and construct a HuggingFace datasets compatible folder.

## Model fine-tuning

Given a dataset created in the previous step, you can then launch training with:
```
cd model_training_and_quantization
MODEL_DIR=<Model directory path> python3 train_model.py
```

Within `train_model.py`, you can select the model that you wish to use to fine-tune, and adjust training epochs and other training parameters.

## Model quantization

After fine-tuning the model, or when using a pretrained model, you can quantize the model into 4-bit format using GPTQ with the following script. Simply run:
```
cd model_training_and_quantization
MODEL_DIR=<Model directory path> python3 quantize_model.py
```

Within `quantize_model.py`, you can select the model that you wish to quantize.