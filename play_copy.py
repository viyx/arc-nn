import numpy as np
import torchvision
import torch
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

from mingpt.utils import set_seed
set_seed(42)



from datasets import CopyDataset

ds_train_params = dict(
    n_episodes=1,
    colors=2,
    fig_w=2,
    fig_h=2,
    field_w=10,
    field_h=10,
    n_figs_on_field=3
)

ds_test_params = dict(
    n_episodes=1,
    colors=2,
    fig_w=3,
    fig_h=3,
    field_w=10,
    field_h=10,
    n_figs_on_field=2
)

train_data = CopyDataset(1000_000, **ds_train_params)
test_data = CopyDataset(10_000, **ds_test_params)



test_data.ds[100,0]



train_data.ds[100,0]



from torch.utils.data import Dataset

class ImageCopyDataset(Dataset):
    """
    wrap up the pytorch CIFAR-10 dataset into our own, which will convert images into sequences of integers
    """
    
    def __init__(self, ds):
        self.ds = ds.ds
        self.vocab_size = ds.colors
        self.block_size = ds.field_w * ds.field_h * 2 - 1 #duplicate field
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx][0].flatten() #take only first episode
        perm_context = x #this context is permanent for all seqs
        not_perm_context = x[:-1] #this context will be masked in attention
        y = x
        x = np.concatenate((perm_context, not_perm_context))
        return x, y # always just predict the next one in the sequence
    
train_dataset = ImageCopyDataset(train_data)
test_dataset = ImageCopyDataset(test_data)



train_dataset[0]



from mingpt.model import GPT, GPTConfig, GPT1Config
from mingpt.trainer import Trainer, TrainerConfig

# we'll do something a bit smaller
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  embd_pdrop=0.0, resid_pdrop=0.1, attn_pdrop=0.1,
                  n_layer=12, n_head=8, n_embd=256)
model = GPT(mconf)

tokens_per_epoch = len(train_dataset) * train_dataset.block_size
train_epochs = 1 # todo run a bigger model and longer, this is tiny

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=128, learning_rate=3e-3,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch, final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='cifar10_model.pt',
                      num_workers=4, tolerance_lim=10000)

trainer = Trainer(model, train_dataset, test_dataset, tconf)



trainer.train()



# checkpoint = torch.load('cifar10_model.pt')
# model.load_state_dict(checkpoint)


