import args_parse


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
    lr=1e-2,
    num_epochs=1,
    num_cores=1,
    num_workers=1,
    weight_decay=0,
    beta1=0.9,
    beta2=0.95,
    opts=MODEL_OPTS.items()
    )

import download
from datasets import MedianDataset, GPTDataset
from torch.utils.data import Dataset, DataLoader
median_dataset = MedianDataset()
train_dataset = GPTDataset(median_dataset.get_train(), n_colors=10, n_context=2048, padding=True)
train_loader = DataLoader(train_dataset, num_workers=FLAGS.num_workers)

# construct a GPT model
from mingpt.model import GPT
model = GPT(FLAGS)

# construct a trainer
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from mingpt.lr_decay import LearningRateDecayCallback

# scheduler
tokens_per_epoch = len(train_dataset) * FLAGS.target_length

lr_decay = LearningRateDecayCallback(learning_rate=FLAGS.lr, warmup_tokens=tokens_per_epoch,
                                    final_tokens=FLAGS.num_epochs*tokens_per_epoch)

trainer = Trainer(tpu_cores=FLAGS.num_cores, precision=16,
                    max_epochs=FLAGS.num_epochs,
                    gradient_clip_val=1.0,
                    callbacks=[lr_decay],
                    progress_bar_refresh_rate=1)

# trainer = Trainer(tpu_cores=1FLAGS.num_cores, precision=16, max_epochs=1,
#                   gradient_clip_val=1.0, 
#                   callbacks=[lr_decay], 
#                   progress_bar_refresh_rate=1)
                #   row_log_interval=1)

# import pdb, sys
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/utilities/xla_device_utils.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/trainer.py')
# sys.path.append('/opt/conda/lib/python3.7/site-packages/pytorch_lightning/accelerators/tpu_accelerator.py')
# pdb.set_trace()

trainer.fit(model, train_loader)

