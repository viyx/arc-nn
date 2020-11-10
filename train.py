import args_parse


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

FLAGS = args_parse.parse_common_options(
    datadir='data/datasets/median/',
    # logdir='/tmp/tensorboard/',
    batch_size=1,
    momentum=0.5,
    lr=1e-2,
    num_epochs=1,
    num_cores=1,
    num_workers=8,
    weight_decay=0,
    beta1=0.9,
    beta2=0.95,
    # opts=MODEL_OPTS.items()
    )


# import pdb, sys
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/utilities/xla_device_utils.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/trainer.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/accelerators/tpu_accelerator.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/utilities/xla_device_utils.py')
# pdb.set_trace()

import numpy as np
from datasets import MedianDataset, GPTDataset
from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.model import GPT


def main():
    pl.seed_everything(333)
    median_dataset = MedianDataset()
    train_dataset = median_dataset.get_train()

    # create val dataset
    # take last 5 task index
    last_five = train_dataset.aug_tasks.iloc[-5].name[0].left
    # take 20% of first (400 - 5) tasks
    twenty = np.random.choice(last_five, int(last_five *.2), replace=False)
    # concat indices for val
    val_ind = np.concatenate([np.arange(last_five, len(train_dataset)), twenty])
    train_ind = np.setdiff1d(np.arange(len(train_dataset)), val_ind)


    train_ds = GPTDataset(Subset(train_dataset, train_ind), n_colors=10, n_context=2048, padding=True)
    val_ds = GPTDataset(Subset(train_dataset, val_ind), n_colors=10, n_context=2048, padding=True)
    test_ds = GPTDataset(median_dataset.get_test(), n_colors=10, n_context=2048, padding=True)

    train_loader = DataLoader(train_ds, num_workers=FLAGS.num_workers)
    val_loader = DataLoader(val_ds, num_workers=FLAGS.num_workers)
    test_loader = DataLoader(test_ds, num_workers=FLAGS.num_workers)

    # construct a GPT model
    model = GPT()

    # scheduler
    # tokens_per_epoch = len(train_dataset) * FLAGS.target_length

    # lr_decay = LearningRateDecayCallback(learning_rate=FLAGS.lr, warmup_tokens=tokens_per_epoch,
                                        # final_tokens=FLAGS.num_epochs*tokens_per_epoch)

    trainer = Trainer(tpu_cores=FLAGS.num_cores,
                        precision=32,
                        max_epochs=2,
                        # fast_dev_run=True,
                        limit_train_batches=0.2,
                        limit_val_batches=100,
                        val_check_interval=10000,
                        # gradient_clip_val=1.0,
                        # callbacks=[lr_decay],
                        progress_bar_refresh_rate=1)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(test_dataloaders=test_loader)


if __name__=='__main__':
    # print(123)
    # Trainer(tpu_cores=1)
    main()

