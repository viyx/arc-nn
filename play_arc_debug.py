from datasets import ARCDataset, ColorPermutation
from itertools import permutations
import numpy as np
import pandas as pd
from copy import copy
import pickle

from torch.utils.data import Dataset

class GPTDataset(Dataset):
    """Flat 2D samples and add specials tokens.
    
    General scheme:
    
    flatten(x) + `promt` + flatten(y) + `end_episode`
    
    Here `flatten` is:
    flat 2D array and add `end_line` in the end of every line.
    """
    
    def __init__(self, ds, n_colors, n_context, padding=False):
        self.ds = ds
        self.n_colors = n_colors
        self.n_context = n_context
        self.padding = padding # expand x to n_context
        self.max_y_size = 30 * 30 + 30 + 1 # max flatten y size with special tokens(end_lines, end_episode)
        
        # make special tokens 
        self.end_line = n_colors + 0    # array of shape (10, 3) has 10 end_lines
        self.promt = n_colors + 1       # promt after every x
        self.end_episode = n_colors + 2 # end of episode
        self.pad = n_colors + 3       # padding token (expand to n_context from the begining)
        
        self.vocab_size = n_colors + 4
        
    def __len__(self):
        return len(self.ds)
    
    def flat_all_sample(self, x, y, x_test, y_test):
        
        def flat_2D_field(x):
            "Add column of `end_line` tokens and flat 2D array."
            a = np.array([[self.end_line]] * x.shape[0], dtype=np.long)
            a = np.hstack([x, a])
            return a.ravel()
    
        def flat_pair(pair):
            "Flat x,y pairs and add `promt` and `end_episode` tokens"
            x, y = pair
            x = flat_2D_field(x)
            y = flat_2D_field(y)
            return np.concatenate([x, self.promt, y, self.end_episode], axis=None)
        
        def pad(seq, to, direct):
            "Pad sequence to left ot right."
            x = np.array([self.pad] * to, dtype=np.long)
            if(direct == 'left'):
                x[-len(seq):] = seq
            if(direct == 'right'):
                x[:len(seq)] = seq
            return x
        
        
        # flat train pairs
        xy = list(map(flat_pair, zip(x, y)))
        xy = np.concatenate(xy, axis=None)
        
        #flat test pair
        
        # take only the first test episode (may be >1)
        # if we keep all test episodes batching would be harder to control
#         if(len(x_test) > 1):
        x_test = x_test[0]
        y_test = y_test[0]
            
        xt = flat_2D_field(x_test)
        yt = flat_2D_field(y_test)
        
        # just add end of episode
        y = np.concatenate([yt, self.end_episode], axis=None)
        
        # pad y to max flattened 2D field
        max_tokens_on_field = 30**2 + 30 + 1
        if(len(y) < max_tokens_on_field and self.padding):
            y = pad(y, max_tokens_on_field, 'right')
        
        # context: concat all
        x = np.concatenate([xy, xt, self.promt, y], axis=None)
        
        # padding
        if(len(x) < self.n_context and self.padding): # expand sample to n_context
            x = pad(x, self.n_context, 'left')
            
        return x, y
    
    def __getitem__(self, id):
        x, y, x_test, y_test = self.ds[id]
        x, y = self.flat_all_sample(x, y, x_test, y_test)
        return x, y


# %%
with open('ds_train_median.pickle', 'rb') as f:
    ds_train_median = pickle.load(f)
    
with open('ds_test_median.pickle', 'rb') as f:
    ds_test_median = pickle.load(f)


# %%
train_dataset = GPTDataset(ds_train_median, 10, 2048, padding=True)
test_dataset = GPTDataset(ds_test_median, 10, 2048, padding=True)


# %%
print(len(train_dataset), len(test_dataset))


# %%
# train_dataset.ds.aug_tasks.loc[(1, slice(None)),][0].values[0][14][1][0].dtype


# %%
train_dataset[1][0].shape


# %%
from mingpt.model import GPT, GPTConfig, GPT1Config
from mingpt.trainer import Trainer, TrainerConfig

# we'll do something a bit smaller
mconf = GPTConfig(train_dataset.vocab_size, block_size=train_dataset.n_context,
                  masked_length = 30 ** 2 + 30 + 1, padding_idx=13,
                  embd_pdrop=0.0, resid_pdrop=0.1, attn_pdrop=0.1,
                  n_layer=12, n_head=8, n_embd=8)
model = GPT(mconf)

tokens_per_epoch = len(train_dataset) * 576 #mean x length
train_epochs = 1 # todo run a bigger model and longer, this is tiny

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=train_epochs, batch_size=1, learning_rate=3e-3,
                      betas = (0.9, 0.95), weight_decay=0,
                      lr_decay=True, warmup_tokens=tokens_per_epoch,
                      final_tokens=train_epochs*tokens_per_epoch,
                      ckpt_path='model.pt', tolerance_lim = 1000,
                      num_workers=1, early_stopping=1000)

trainer = Trainer(model, train_dataset, test_dataset, tconf)


# %%
trainer.train()


# %%



