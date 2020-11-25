# import args_parse

# MODEL_OPTS = {
#     'vocab_size': 14,
#     'block_size': 2018,
#     'target_length': 30 ** 2 + 30 + 1,
#     'padding_index': 13,
#     'embd_pdrop': 0.0,
#     'resid_pdrop': 0.1,
#     'attn_pdrop': 0.1,
#     'n_layer': 2,
#     'n_head': 8,
#     'n_embd': 8
# }

# MODEL_OPTS = {
#     '--vocab_size': {
#         'type': int,
#         'default': 14,
#     },
#     '--block_size': {
#         'type': int,
#         'default': 2048,
#     },
#     '--target_length': {
#         'type': int,
#         'default': 30 ** 2 + 30 + 1,  #squared max shape + 30 endlines + 1 end episode
#     },
#     '--padding_idx': {
#         'type': int,
#         'default': 13,
#     },
#     '--resid_pdrop': {
#         'type': float,
#         'default': 0.1,
#     },
#     '--attn_pdrop': {
#         'type': float,
#         'default': 0.1,
#     },
#     '--embd_pdrop': {
#         'type': float,
#         'default': 0.0,
#     },
#     '--n_layer': {
#         'type': int,
#         'default': 2,
#     },
#     '--n_embd': {
#         'type': int,
#         'default': 8,
#     },
#     '--n_head': {
#         'type': int,
#         'default': 8,
#     },
# }

# FLAGS = args_parse.parse_common_options(
#     datadir='data/datasets/median/',
#     logdir='data/logs/tensorboard/',
#     batch_size=8,
#     momentum=0.5,
#     lr=0.01,
#     num_epochs=5,
#     num_cores=1,
#     num_workers=1,
#     log_steps=50,
#     val_steps=15000,
#     opts=MODEL_OPTS.items()
#     )


import sys
import os
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
import download
import logging
import numpy as np
from datasets import MaxNDataset, GPTDataset, LightWeightColorPermutation
import pickle
from utils import set_seed
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
from mingpt.model import GPT

# set up logging
# logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
# )

logger = logging.getLogger(__name__)


def _train_update(suffix, step, loss, tracker, epoch, writer):
    writer.add_scalar(suffix + '/train', loss, step)


def get_dataset(fast_run=False):
        if(fast_run):
            train_trans = None
            test_trans = None
            val_trans = None
        else:
            train_trans = [LightWeightColorPermutation(max_colors=FLAGS.n_colors, limit=10000)]
            test_trans = None
            val_trans = [LightWeightColorPermutation(max_colors=FLAGS.n_colors, limit=100)]


        ds = MaxNDataset(
            transforms=[train_trans, test_trans, val_trans],
            **FLAGS.__dict__)
        train, test, val = ds.datasets
        logger.info("Dataset lenghts: train = {}, test = {}, val = {}".format(len(train), len(test), len(val)))
        return train, test, val


def train(rank):
    set_seed(SEED)
    global SERIAL_EXEC, MODEL
    train, test, val = SERIAL_EXEC.run(lambda: get_dataset(FLAGS.fast_run))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=SEED)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
        seed=SEED)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
        seed=SEED)
  
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        num_workers=FLAGS.num_workers,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=FLAGS.batch_size,
        sampler=val_sampler,
        num_workers=FLAGS.num_workers,
        drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=FLAGS.batch_size,
        sampler=test_sampler,
        num_workers=FLAGS.num_workers,
        drop_last=False)

    device = xm.xla_device()
    model = MODEL.to(device)
    writer = None
    if xm.is_master_ordinal():
        writer = SummaryWriter()
        FLAGS.log_dir = writer.log_dir

    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=FLAGS.lr,
    #     momentum=FLAGS.momentum,
    #     weight_decay=1e-4)

    FLAGS.lr *= xm.xrt_world_size()
    optimizer = model.configure_optimizers(FLAGS)

    def train_loop_fn(loader, epoch):
        # return 100, [23, 32, 333]
        tracker = xm.RateTracker()
        model.train()
        tq = tqdm(loader, total=int(len(loader))) if rank == 0 else loader
        loss_mean = 0.0
        loss_vals = []
        for step, (data, target) in enumerate(tq):
            optimizer.zero_grad()
            _, loss = model(data, target)
            loss = loss.mean()
            loss.backward()
            loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)
            if step % FLAGS.log_steps == 0  or step == len(loader) - 1:
                loss_mr = xm.mesh_reduce('loss', loss_mean, np.mean)
                if(rank == 0):
                    tq.set_description(f"epoch {epoch}: train loss {loss_mr:.5f}")
                    xm.add_step_closure(_train_update, args=('train', step, loss_mr, None, epoch, writer))
            if not step == 0 and step % FLAGS.val_steps == 0 or step == len(tq) - 1:
                val_device_loader = pl.MpDeviceLoader(val_loader, device)
                val_loss = val_loop_fn(val_device_loader)
                loss_vals.append(val_loss)
                model.train()
        return loss_mr, loss_vals

    def val_loop_fn(loader):
        model.eval()
        tracker = xm.RateTracker()
        loss_mean = 0.0
        tq = tqdm(loader, total=int(len(loader))) if rank == 0 else loader
        with torch.no_grad():
            for step, (data, target) in enumerate(tq):
                _, loss = model(data, target)
                loss = loss.mean()
                loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
                tracker.add(FLAGS.batch_size)
                if step % FLAGS.log_steps == 0 or step == len(loader) - 1:
                    loss_mr = xm.mesh_reduce('loss_val', loss_mean, np.mean)
                    if(xm.is_master_ordinal()):
                        tq.set_description(f"epoch {epoch}: val loss {loss_mr:.5f}")
                        xm.add_step_closure(_train_update, args=('val', step, loss_mr, None, epoch, writer))
        return loss_mr

    def test_loop_fn(loader):
        model.eval()
        tracker = xm.RateTracker()
        loss_mean = 0.0
        tq = tqdm(loader, total=int(len(loader))) if xm.is_master_ordinal() else loader
        for step, (data, target) in enumerate(tq):
            _, loss = model(data, target)
            loss = loss.mean()
            loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
            tracker.add(FLAGS.batch_size)
            loss_mr = xm.mesh_reduce('loss_test', loss_mean, np.mean)
            if(xm.is_master_ordinal()):
                tq.set_description(f"epoch {epoch}: test loss {loss_mr:.5f}")
                xm.add_step_closure(_train_update, args=('test', epoch, loss_mr, None, epoch, writer))
        return loss_mr
    
    if(xm.is_master_ordinal()):
        writer.add_hparams(FLAGS.__dict__, {'test':123})
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    test_device_loader = pl.MpDeviceLoader(test_loader, device)
    for epoch in range(1, FLAGS.num_epochs + 1):
        xm.master_print('epoch {}: train begin {}'.format(epoch, test_utils.now()))
        train_sampler.set_epoch(epoch)
        train_loss, val_losses = train_loop_fn(train_device_loader, epoch)
        test_loss = test_loop_fn(test_device_loader)
        curr_state = {
            'model_state_dict': MODEL._model.cpu().state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
        if(xm.is_master_ordinal()):
            cpu_data = xm._maybe_convert_to_cpu(curr_state, convert=True)
            torch.save(cpu_data, os.path.join(FLAGS.log_dir, f'model_ep{epoch}.pt'))
            metrics = {
                'tr_loss':train_loss,
                'val_loss':np.mean(val_losses),
                'test_loss': test_loss}
            params = {
                'lr': FLAGS.lr,
                'epoch':epoch,
                'batch_size': FLAGS.batch_size,
                'num_cores': FLAGS.num_cores,
                }
            writer.add_hparams(params, metrics) 
        xm.rendezvous('torch_xla.core.xla_model.save')
        
        xm.master_print('epoch {} train end {}, train: {:.2f}, val: {:.2f}, test: {:.2f}'.
            format(epoch, test_utils.now(), train_loss, np.mean(val_losses), test_loss))
        MODEL.to(device)
    if(xm.is_master_ordinal()):
        writer.flush()
        writer.close()
    
def add_train_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_cores", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=15000)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--fast_run", action='store_true', default=False)
    return parser


parser = add_train_args(ArgumentParser())
parser = GPT.add_model_specific_args(parser)
parser = MaxNDataset.add_data_specific_args(parser)
FLAGS = parser.parse_args()
# FLAGS.betas = eval(FLAGS.betas)
# FLAGS.split = eval(FLAGS.split)
# chkp = torch.load('model.pt', map_location='cpu')
m = GPT(FLAGS)
# m.load_state_dict(chkp['model_state_dict'])
MODEL = xmp.MpModelWrapper(m)
SERIAL_EXEC = xmp.MpSerialExecutor()
SEED = 333

def map_fn(rank, args):
    global FLAGS
    FLAGS = args
    train(rank)
    # sys.exit(21)

if __name__ == '__main__':
    os.environ['XRT_TPU_CONFIG'] = "tpu_worker;0;10.185.7.138:8470"
    os.environ['PYTHONWARNINGS'] = "ignore:semaphore_tracker:UserWarning"
    os.environ['XLA_USE_BF16'] = "1"
    xmp.spawn(map_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)