import os
import requests
import logging


BUCKET_ADDR = 'https://storage.googleapis.com/viy_data/'

logger = logging.getLogger(__name__)

def load_from_bucket(datadir, files):
    "Download files from google cloud if not exist."
    for filename in files:
        file = os.path.join(datadir, filename)
        os.makedirs(datadir, exist_ok=True)
        if not os.path.exists(file):
            url = BUCKET_ADDR + file
            r = requests.get(url)
            with open(file, 'wb') as desc:
                desc.write(r.content)
        else:  logger.info(f'skip download: file `{file}` exists')

