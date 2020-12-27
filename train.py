import sys
import os
from tqdm import tqdm
from datetime import datetime
# from argparse import ArgumentParser
import download
import logging
import numpy as np
from datasets import MaxNDataset, GPTDataset, ColorPermutation
# import pickle
from utils import set_seed
import torch
import torch.optim as optim
from torch.utils.data import Subset
# from torch.utils.tensorboard import SummaryWriter
# if('XRT_TOU_CONFIG' in os.environ):
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
# import torch_xla.test.test_utils as test_utils
from mingpt.model import GPT
import wandb
from wandb.util import generate_id
import hydra
# from hydra.core.utils import setup_globals
from omegaconf import DictConfig, OmegaConf


LOGGER = logging.getLogger(__name__)


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
        LOGGER.info("Dataset lenghts: train = {}, test = {}, val = {}".format(len(train), len(test), len(val)))
        return train, test, val


def train(rank):
    def train_loop_fn(loader, val_loader, epoch):
        MODEL.train()
        tq = tqdm(loader, unit_scale=FLAGS.n_cores) if xm.is_master_ordinal(True) else loader
        loss_mean = 0.0
        # val_losses = []
        for step, (x, y) in enumerate(tq):
            optimizer.zero_grad()
            _, loss = MODEL(x, y)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(MODEL.parameters(), FLAGS.grad_norm_clip)
            loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
            xm.optimizer_step(optimizer)

            # log
            if (step*FLAGS.n_cores) % FLAGS.log_step < FLAGS.n_cores or step == len(loader) - 1:
                loss_mean_mr = xm.mesh_reduce('loss', loss_mean, np.mean)
                if(xm.is_master_ordinal(True)):
                    tq.set_description(f"epoch {epoch}: loss: {loss_mean_mr:.3f}")
                    wandb.log({'loss_train':loss_mean_mr}, step=step*FLAGS.n_cores)

        # validate on the end of epoch or per interval
        # if not step == 0 and (step*FLAGS.n_cores) % FLAGS.val_step < FLAGS.n_cores or step == len(tq) - 1:
        # if(FLAGS.save):
            # save(epoch, step*FLAGS.n_cores)
        val_loss = val_loop_fn(val_loader)
        # val_losses.append(val_loss)
        if(xm.is_master_ordinal()):
            wandb.log({'loss_val':val_loss}, step=step*FLAGS.n_cores)
        MODEL.train()
        return loss_mean, val_loss

    def val_loop_fn(loader):
        MODEL.eval()
        loss_mean = 0.0
        tq = tqdm(loader, leave=False, unit_scale=FLAGS.n_cores) if xm.is_master_ordinal(True) else loader
        with torch.no_grad():
            for step, (data, target) in enumerate(tq):
                _, loss = MODEL(data, target)
                loss = loss.mean()
                loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
                if(xm.is_master_ordinal()):
                    tq.set_description(f"epoch {epoch}: val loss {loss_mean:.3f}")
            loss_mr = xm.mesh_reduce('loss_val', loss_mean, np.mean)
        return loss_mr

    def test_loop_fn(loader):
        MODEL.eval()
        loss_mean = 0.0
        tq = tqdm(loader, total=int(len(loader))) if xm.is_master_ordinal(True) else loader
        with torch.no_grad():
            for step, (data, target) in enumerate(tq):
                _, loss = MODEL(data, target)
                loss = loss.mean()
                loss_mean = (loss_mean * (step) + loss.item())/(step + 1)
                loss_mr = xm.mesh_reduce('loss_test', loss_mean, np.mean)
        return loss_mr
    
    def save(filename):
        # copied from xm.save()
        # FLAGS will be global after implementing distributed mode
        master_only=True
        global_master=False
        should_write_data = not master_only or xm.is_master_ordinal(local=not global_master)
        curr_state = {
            'model_state_dict': xm._maybe_convert_to_cpu(MODEL.state_dict(), convert=should_write_data),
            'optimizer_state_dict': xm._maybe_convert_to_cpu(optimizer.state_dict(), convert=should_write_data),
            'flags': FLAGS._content,
        }
    
        if(xm.is_master_ordinal()):
            torch.save(curr_state, filename)
            wandb.save(filename)
        xm.rendezvous('save')

    set_seed(FLAGS.seed)
    train, test, val = get_dataset(FLAGS.fast_run)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=FLAGS.seed)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
        seed=FLAGS.seed)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
        seed=FLAGS.seed)
  
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
    MODEL.to(device)
    optimizer = MODEL.configure_optimizers(FLAGS)

    if(FLAGS.scale_lr):
        FLAGS.lr *=FLAGS.n_cores

    # resume
    if xm.is_master_ordinal(True):
        wandb.init(
            id=HYDRA_FLAGS.job.id,
            group=HYDRA_FLAGS.job.name,
            project='gpt',
            config=FLAGS._content,
            resume='allow',
            settings=wandb.Settings(_disable_stats=True))

        if wandb.run.resumed:
            api = wandb.Api()
            run = api.run(f'viy/gpt/{HYDRA_FLAGS.job.id}')

            # check file to restore
            can_restore = False
            for f in run.files():
                if FLAGS.preempt in f.name:
                    can_restore = True
            
            if(can_restore):            
                wandb.restore(FLAGS.preempt, root='./')
                chpt = torch.load(FLAGS.preempt)
                MODEL.load_state_dict(chpt['model_state_dict'])
                optimizer.load_state_dict(chpt['optimizer_state_dict'])
                # FLAGS.update(chpt['flags'])
                OmegaConf.update(FLAGS, 'patience', chpt['flags']['patience'])
                OmegaConf.update(FLAGS, 'init_epoch', chpt['flags']['init_epoch'])
                OmegaConf.update(FLAGS, 'best', chpt['flags']['best'])
                LOGGER.info(f'Load weights from checkpoint {FLAGS.preempt}')
    
    if(FLAGS.n_cores > 1):
        train_loader = pl.MpDeviceLoader(train_loader, device)
        val_loader = pl.MpDeviceLoader(train_loader, device)
        test_loader = pl.MpDeviceLoader(test_loader, device)

    xm.rendezvous('wait all')
    for epoch in range(FLAGS.init_epoch, FLAGS.n_epochs + 1):
        if xm.is_master_ordinal():
            LOGGER.info('Start epoch {} with batch_size {} and best {}'.
                format(epoch, FLAGS.batch_size, FLAGS.best))

        train_loss, val_loss = train_loop_fn(train_loader, val_loader, epoch)
        FLAGS.init_epoch += 1
        if(FLAGS.best): 
            if FLAGS.best <= val_loss:
                FLAGS.patience += 1
            else: 
                FLAGS.best = val_loss
                FLAGS.patience = 0
                wandb.run.summary["best_loss"] = val_loss
                wandb.run.summary["best_epoch"] = epoch
                # save best checkpoint independently of preemptible checkpoint
                save('best.pt')

        else: FLAGS.best = val_loss
        save('preempt.pt')

        test_loss = test_loop_fn(test_loader)

        if xm.is_master_ordinal():
            LOGGER.info('Finish epoch {}. train: {:.2f}, val: {:.2f}, test: {:.2f}, best: {:.2F}, patience: {}/{}'.
                format(epoch, train_loss, val_loss, test_loss, FLAGS.best, FLAGS.patience, FLAGS.early_stop_patience))
        
        if(FLAGS.patience == FLAGS.early_stop_patience):
            LOGGER.info('Stop training')
            break
                
    if(xm.is_master_ordinal(local=True)):
        wandb.finish()


def prepare_flags_for_training():
    # global FLAGS

    # exlude validation per interval
    if FLAGS.val_step == 0:
        FLAGS.val_step = np.inf

    # keep only dict from DictConfig
    # FLAGS = FLAGS._content


def map_fn(rank, *args):
    global FLAGS, HYDRA_FLAGS, MODEL
    FLAGS, HYDRA_FLAGS = args
    prepare_flags_for_training()
    MODEL = GPT(FLAGS)
    wandb.login()
    train(rank)
    xm.rendezvous('exit')
    # sys.exit(21)

@hydra.main(config_path='conf', config_name="config")
def main(cfg: DictConfig):
    assert os.environ['WANDB_API_KEY'], 'Specify wandb api key'

    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ['XLA_USE_BF16'] = '1'
    if(cfg.debug):
        os.environ['XRT_DEVICE_MAP'] = 'CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0'
        os.environ['XRT_WORKERS'] = 'localservice:0;grpc://localhost:40934'
    else:
        assert os.environ['XRT_TPU_CONFIG'], 'Specify xla device.'
    
    # allow to change
    # OmegaConf.set_struct(cfg, False)
    full_conf = hydra.core.hydra_config.HydraConfig.get()

    xmp.spawn(map_fn, args=(cfg, full_conf,), nprocs=cfg.n_cores, start_method='spawn')


if __name__ == '__main__':
    OmegaConf.register_resolver("wandb_id", lambda: generate_id())
    main()

