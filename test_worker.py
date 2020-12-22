from rq import Connection, Worker

import sys
import os
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import download
import logging
import numpy as np
from datasets import MaxNDataset, GPTDataset, ColorPermutation
import pickle
from utils import set_seed
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
from mingpt.model import GPT
import wandb



with Connection():
    qs = ['default']

    w = Worker(qs)
    w.work()