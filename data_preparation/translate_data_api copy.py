import os, json
import requests
import ast, logging, time

logger = logging.getLogger()
logger.setLevel(logging.INFO)


DIR_NAME = os.getenv('DIR_NAME')

endpoint = 'https://translation.googleapis.com/language/translate/v2'
API_KEY = os.getenv('API_KEY')
LANGUAGE = os.getenv('LANGUAGE')

filenames = [
    os.path.join(DIR_NAME,x)
    for x in os.listdir(DIR_NAME)
    if '.json' in x
]

dirName = 'API_'+LANGUAGE+'_'+DIR_NAME
os.makedirs(dirName, exist_ok=True)
allFiles = {}

for indivFile in filenames:
    logger.info('running for {}'.format(indivFile))
    data = json.loads(open(indivFile).read())
    responses = ast.literal_eval(data['choices'][0]['message']['content'].strip())
    splitResp = [x.split('|') for x in responses]
    allFiles[indivFile] = splitResp

for indivFile in filenames:
    newResp = []
    splitResp = allFiles[indivFile]
    for ans in splitResp:
        try:
            body = {
                'q': ans[1].lower().split('text:')[1],
                'source': 'en',
                'target': LANGUAGE,
                'format': 'text'
            }
            res = requests.post(endpoint,
                json=body,
                headers={
                    'X-goog-api-key': os.getenv('API_KEY')
                })
            output = res.json()
            output = output['data']['translations'][0]['translatedText']
            newResp.append([ans[0],' Text: '+output])
        except Exception as e:
            logger.error(e)
        time.sleep(0.05)
    newResp = ['|'.join(x) for x in newResp]
    fileName = os.path.basename(indivFile)
    open(os.path.join(dirName,fileName),'w').write(json.dumps(newResp))
