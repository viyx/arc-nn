from rq import Connection, Worker

import sys
import os
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import logging
import numpy as np
# from datasets import MaxNDataset, GPTDataset, ColorPermutation
import pickle
# from utils import set_seed
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
# from ... import mingpt
import wandb


os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;10.42.117.58:8470'

with Connection():
    qs = ['default']

    w = Worker(qs, )
    w.work(max_jobs=1, burst=True, logging_level='DEBUG')
