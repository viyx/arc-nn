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
import hydra


def get_dataset(fast_run):
        if(fast_run):
            train_trans = None
            test_trans = None
            val_trans = None
        else:
            train_trans = [ColorPermutation(max_colors=FLAGS.n_colors, limit=1000)]
            test_trans = None
            val_trans = [ColorPermutation(max_colors=FLAGS.n_colors, limit=100)]
        ds = MaxNDataset(transforms=[train_trans, test_trans, val_trans], config=FLAGS)
        train, test, val = ds.datasets
        logger.info("Dataset lenghts: train = {}, test = {}, val = {}".format(len(train), len(test), len(val)))
        return train, test, val


def train(rank):
    def train_loop_fn(loader, val_loader, epoch):
        model.train()
        tq = tqdm(loader, unit_scale=FLAGS.n_cores) if xm.is_master_ordinal(True) else loader
        loss_mean = 0.0
        val_losses = []
        for step, (x, y) in enumerate(tq):
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_norm_clip)
            loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
            xm.optimizer_step(optimizer)

            # log
            if (step*FLAGS.n_cores) % FLAGS.log_steps < FLAGS.n_cores or step == len(loader) - 1:
                loss_mean_mr = xm.mesh_reduce('loss', loss_mean, np.mean)
                if(xm.is_master_ordinal(True)):
                    tq.set_description(f"epoch {epoch}: loss: {loss_mean_mr:.3f}")
                    wandb.log({'loss_train':loss_mean_mr}, step=step*FLAGS.n_cores)

            # validate
            if not step == 0 and (step*FLAGS.n_cores) % FLAGS.val_steps < FLAGS.n_cores or step == len(tq) - 1:
                if(FLAGS.save):
                    # save
                    save(epoch, step*FLAGS.n_cores)
                val_loss = val_loop_fn(val_loader)
                val_losses.append(val_loss)
                if(xm.is_master_ordinal()):
                    wandb.log({'loss_val':val_loss}, step=step*FLAGS.n_cores)
                model.train()
        return loss_mean, val_losses

    def val_loop_fn(loader):
        model.eval()
        loss_mean = 0.0
        tq = tqdm(loader, leave=False, unit_scale=FLAGS.n_cores) if xm.is_master_ordinal(True) else loader
        with torch.no_grad():
            for step, (data, target) in enumerate(tq):
                _, loss = model(data, target)
                loss = loss.mean()
                loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
                if(xm.is_master_ordinal()):
                    tq.set_description(f"epoch {epoch}: val loss {loss_mean:.3f}")
            loss_mr = xm.mesh_reduce('loss_val', loss_mean, np.mean)
        return loss_mr

    def test_loop_fn(loader):
        model.eval()
        loss_mean = 0.0
        tq = tqdm(loader, total=int(len(loader))) if xm.is_master_ordinal(True) else loader
        with torch.no_grad():
            for step, (data, target) in enumerate(tq):
                _, loss = model(data, target)
                loss = loss.mean()
                loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
                loss_mr = xm.mesh_reduce('loss_test', loss_mean, np.mean)
        return loss_mr
    
    def save(epoch, step):
        # copied from xm.save()
        # flags will be global after implementing distributed mode
        master_only=True
        global_master=False
        should_write_data = not master_only or xm.is_master_ordinal(local=not global_master)
        curr_state = {
            'model_state_dict': xm._maybe_convert_to_cpu(MODEL._model.state_dict(), convert=should_write_data),
            'optimizer_state_dict': xm._maybe_convert_to_cpu(optimizer.state_dict(), convert=should_write_data),
            'FLAGS': FLAGS
        }
    
        if(xm.is_master_ordinal()):
            name = os.path.join(wandb.run.dir, f'model_epoch{epoch}_step{step}.pt')
            torch.save(curr_state, name)
            wandb.save(name, base_path=wandb.run.dir)
        xm.rendezvous('save')

    set_seed(SEED)
    # global SERIAL_EXEC, MODEL
    global MODEL
    # train, test, val = SERIAL_EXEC.run(lambda: get_dataset(FLAGS.fast_run))
    train, test, val = get_dataset(FLAGS.fast_run)

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
        num_workers=FLAGS.n_workers,
        drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=FLAGS.batch_size,
        sampler=val_sampler,
        num_workers=FLAGS.n_workers,
        drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=FLAGS.batch_size,
        sampler=test_sampler,
        num_workers=FLAGS.n_workers,
        drop_last=False)

    if(FLAGS.debug):
        device = torch.device('cpu')
        # device = xm.xla_device(devkind='CPU')
    else:
        device = xm.xla_device()
    model = MODEL.to(device)
    writer = None
    
    if(FLAGS.scale_lr):
        FLAGS.lr *=FLAGS.n_cores
    optimizer = model.configure_optimizers(FLAGS)

    if xm.is_master_ordinal(True):
        wandb.init(project='test', config=FLAGS, settings=wandb.Settings(_disable_stats=True))

    if(FLAGS.n_cores > 1):
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)

    xm.rendezvous('wait all')
    for epoch in range(1, FLAGS.n_epochs + 1):
        xm.master_print('epoch {}: train begin {}, batch_size {}'.format(epoch, test_utils.now(), FLAGS.batch_size))
        # train_sampler.set_epoch(epoch)
        train_loss, val_losses = train_loop_fn(train_loader, val_loader, epoch)
        test_loss = test_loop_fn(test_loader)
        xm.master_print('epoch {} train end {}, train: {:.2f}, val: {:.2f}, test: {:.2f}'.
            format(epoch, test_utils.now(), train_loss, np.mean(val_losses), test_loss))
    if(xm.is_master_ordinal(local=True)):
        writer.flush()
        writer.close()
    
def add_train_args(parent_parser):
    # TODO check add_positions 
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=10) # log
    parser.add_argument("--val_steps", type=int, default=15000) # make validation every
    parser.add_argument("--log_dir", type=str) # tensorboard and checkpoint dir
    parser.add_argument("--fast_run", action='store_true') # use fast dataset with no transformations
    parser.add_argument("--log_console", action='store_true') # enable logging
    parser.add_argument("--scale_lr", action='store_true') # mult lr by num_cores as sm.optimizer sums batches grads(see https://github.com/pytorch/xla/issues/1781#issuecomment-601849130)
    parser.add_argument("--debug", action="store_true") # work with power on TPU, set device to cpu
    parser.add_argument("--grad_norm_clip", type=float, default=1.0)
    parser.add_argument("--save", action='store_true')
    return parser


parser = add_train_args(ArgumentParser())
parser = MaxNDataset.add_data_specific_args(parser)
parser = GPT.add_model_specific_args(parser)
FLAGS, unknown = parser.parse_known_args()
m = GPT(FLAGS)
MODEL = m
# MODEL = xmp.MpModelWrapper(m)
# SERIAL_EXEC = xmp.MpSerialExecutor()
SEED = 333
logger = logging.getLogger(__name__)

if(FLAGS.log_console):
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.DEBUG,
    )


def map_fn(rank, args):
    global FLAGS
    FLAGS = args
    train(rank)

    # Barrier to prevent master from exiting before workers connect.
    xm.rendezvous('exit')
    # sys.exit(21)

from omegaconf import DictConfig
@hydra.main(config_name="config")
def main(cfg: DictConfig):
    # print(os.environ['XRT_TPU_CONFIG'])
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ['XLA_USE_BF16'] = '1'
    # print(os.environ['XRT_TPU_CONFIG'])
    if(FLAGS.debug):
        print('debug')
        os.environ['XRT_DEVICE_MAP'] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
        os.environ['XRT_WORKERS'] = 'localservice:0;grpc://localhost:40934'
    else:
        # if()
        print(os.environ['XRT_TPU_CONFIG'])
        # os.environ['XRT_TPU_CONFIG'] = "tpu_worker;0;10.78.79.90:8470"
    wandb.login()
    FLAGS.__dict__.update(cfg)
    xmp.spawn(map_fn, args=(FLAGS,), nprocs=FLAGS.n_cores, start_method='spawn')


if __name__ == '__main__':
    # FLAGS.debug = True
    # os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    # os.environ['XLA_USE_BF16'] = '1'
    # # FLAGS.debug = True
    # if(FLAGS.debug):
    #     os.environ['XRT_DEVICE_MAP'] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
    #     os.environ['XRT_WORKERS'] = 'localservice:0;grpc://localhost:40934'
    # else:
    #     os.environ['XRT_TPU_CONFIG'] = "tpu_worker;0;10.78.79.90:8470"
    # xmp.spawn(map_fn, args=(FLAGS,), nprocs=FLAGS.n_cores)
    main()

