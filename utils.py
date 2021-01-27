import random
import numpy as np
import torch
import requests
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_load_and_save_from_bucket_if_not_exist(bucket, datadir, files, logger):
    "Download files from google cloud if they are not exist."
    for filename in files:
        file = os.path.join(datadir, filename)
        os.makedirs(datadir, exist_ok=True)
        if not os.path.exists(file):
            url = bucket + file
            r = requests.get(url)
            with open(file, 'wb') as desc:
                desc.write(r.content)
        else:  logger.info(f'skip download: file `{file}` exists')