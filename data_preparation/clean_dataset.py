from datasets import Dataset
import json
import os, ast
import logging
import sys
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

allDirs = [x for x in os.listdir('./') if os.path.isdir(x) and 'dataset' not in x]

dataRow = []
for folderName in allDirs:
    originalResp = True
    if any([x in folderName for x in ['Chinese','Malay','Tamil','zh','ms','ta']]):
        originalResp = False
    if 'offensive' in folderName:
        label = "offensive"
    elif 'ok' in folderName:
        label = 'ok'
    else:
        logger.error('cannot find label for {}'.format(folderName))
        continue
    logger.info('processing for {}'.format(folderName))
    filesWithin = [os.path.join(folderName,x) for x in os.listdir(folderName)]
    for indivFile in filesWithin:
        try:
            data = json.loads(open(indivFile).read())
            if originalResp:
                data = ast.literal_eval(data['choices'][0]['message']['content'].strip())
            for indivData in data:
                imageData = indivData.split('|')[0].replace('Image:','').strip()
                textData = indivData.split('|')[1].replace('Text:','').replace("'","").strip()
                dataRow.append({
                    'image':imageData,
                    'text':textData,
                    'label':label  
                })
        except Exception as e:
            logger.error('while processing {}, met error: {}'.format(
                indivFile,e
            ))
            raise e

# replace all text in parantheses
for row in dataRow:
    text = row['text']
    newText = re.sub(r'\([^)]*\)', '', text)
    if newText != text:
        logger.info('replacing {} with {}'.format(text,newText))
    row['text'] = newText 

ds = Dataset.from_list(dataRow)
ds.save_to_disk('dataset_10Apr2024')