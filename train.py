import args_parse

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

MODEL_OPTS = {
    '--vocab_size': {
        'type': int,
        'default': 14,
    },
    '--block_size': {
        'type': int,
        'default': 2048,
    },
    '--target_length': {
        'type': int,
        'default': 30 ** 2 + 30 + 1,  #squared max shape + 30 endlines + 1 end episode
    },
    '--padding_idx': {
        'type': int,
        'default': 13,
    },
    '--resid_pdrop': {
        'type': float,
        'default': 0.1,
    },
    '--attn_pdrop': {
        'type': float,
        'default': 0.1,
    },
    '--embd_pdrop': {
        'type': float,
        'default': 0.0,
    },
    '--n_layer': {
        'type': int,
        'default': 2,
    },
    '--n_embd': {
        'type': int,
        'default': 8,
    },
    '--n_head': {
        'type': int,
        'default': 8,
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='data/datasets/median/',
    logdir='/tmp/tensorboard/',
    batch_size=1,
    momentum=0.5,
    lr=0.01,
    num_epochs=1,
    num_cores=1,
    num_workers=1,
    opts=MODEL_OPTS.items()
    )


import sys
import download
import logging
from datasets import MedianDataset, GPTDataset
import pickle
from utils import set_seed
import torch
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
from mingpt.model import GPT


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)


def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)


def train():
    set_seed(333)
    median_dataset = MedianDataset()
    train_dataset = GPTDataset(median_dataset.get_train(), n_colors=10, n_context=2048, padding=True)
    # test_dataset = GPTDataset(median_dataset.get_test(),  n_colors=10, n_context=2048, padding=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
  
    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset,
    #     num_replicas=xm.xrt_world_size(),
    #     rank=xm.get_ordinal(),
    #     shuffle=False)
  
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        num_workers=FLAGS.num_workers,
        drop_last=FLAGS.drop_last)

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=FLAGS['batch_size'],
    #     sampler=test_sampler,
    #     shuffle=False,
    #     num_workers=FLAGS['num_workers'],
    #     drop_last=FLAGS['drop_last'])

    # lr = FLAGS.lr * xm.xrt_world_size()

    if (FLAGS.device):
        device = FLAGS.device
    else:
        device = xm.xla_device()
    model = GPT(FLAGS).to(device).train()
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=1e-4)

    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            _, loss = model(data, target)
            loss = loss.mean()
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)
            # if lr_scheduler:
                # lr_scheduler.step()
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    _train_update, args=(device, step, loss, tracker, epoch, writer))
    
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    # test_device_loader = pl.MpDeviceLoader(test_loader, device)
    # accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        train_loop_fn(train_device_loader, epoch)
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
        # accuracy = test_loop_fn(test_device_loader, epoch)
        # xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
            # epoch, test_utils.now(), accuracy))
        # max_accuracy = max(accuracy, max_accuracy)
        test_utils.write_to_summary(
            writer,
            epoch,
            # dict_to_write={'Accuracy/test': accuracy},
            write_xla_metrics=True)
        # if FLAGS.metrics_debug:
            # xm.master_print(met.metrics_report())

    test_utils.close_summary_writer(writer)
    # xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return

    

def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    train()
    # sys.exit(21)


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)