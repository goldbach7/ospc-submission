import requests
import os, time
import logging
import promptDict
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

endpoint = 'https://api.together.xyz/v1/chat/completions'
MODEL_ID = os.getenv('MODEL_ID')
API_KEY = os.getenv('API_KEY')
PROMPT_KEY = os.getenv('PROMPT_KEY')
NUM_RUNS = int(os.getenv('NUM_RUNS',10))

prompt = promptDict.prompts[PROMPT_KEY]
dirName = os.path.basename(MODEL_ID)+'_'+PROMPT_KEY
os.makedirs(dirName, exist_ok=True)

for i in range(NUM_RUNS):
    time.sleep(1)
    try:
        res = requests.post(endpoint, json={
            "model": MODEL_ID,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1,
            "top_p": 1,
            "top_k": 500,
            "repetition_penalty": 1,
            "stop": [
                "[/INST]",
                "</s>"
            ]
        }, headers={
            "Authorization": "Bearer {}".format(API_KEY),
        })
        output = res.json()
        fileName = '{}.json'.format(i)
        open(os.path.join(dirName,fileName),'w').write(json.dumps(output))
    except Exception as e:
        logger.error(e)
