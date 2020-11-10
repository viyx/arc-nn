"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
# from pytorch_lightning.core.step_result import TrainResult

# logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        # self.masked_length = n_y
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    # n_layer = 1
    # n_head = 4
    # n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask_tril", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # tcontext mask to ensure we dont learn predict context itself
        self.register_buffer("mask_rect", torch.cat([torch.zeros((config.block_size - config.target_length, config.block_size)),
                                    torch.ones((config.target_length, config.block_size))])
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask_tril[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(self.mask_rect[:,:,:T,:T] == 0, 0.0) #hide perm context from backward pass
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        att = self.attn(self.ln1(x))
        x = x + att
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            import argparse
            config = {
                'padding_idx':13,
                'block_size':2048,
                'vocab_size':14,
                'target_length':30**2+30+1,
                'n_embd':8,
                'n_layer':2,
                'n_head':8,
                'embd_pdrop':0.0,
                'attn_pdrop':0.1,
                'resid_pdrop':0.1,
                'lr':1e-2,
                'momentum':0.5
            }
            config = argparse.Namespace(**config)

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = config

        self.padding_idx = self.config.padding_idx
        self.block_size = self.config.block_size
        # positions start from `1` as `0` token reserved as padding index
        self.register_buffer('positions', torch.arange(1,self.block_size+1))
        # input embedding stem
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd, padding_idx=self.padding_idx)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.n_context, config.n_embd))
        self.pos_emb = nn.Embedding(self.config.block_size + 1, self.config.n_embd, padding_idx=0)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    # def get_block_size(self):
    #     return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # create the optimizer
        # no_decay = ["bias", "LayerNorm.weight"]
        # params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        # params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        # optim_groups = [
        #     {"params": params_decay, "weight_decay": self.config.weight_decay},
        #     {"params": params_nodecay, "weight_decay": 0.0},
        # ]
        # optimizer = torch.optim.AdamW(optim_groups, lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=1e-4)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        # position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector

        #set `0` for positions with padding index
        positions = self.positions.masked_fill((idx == self.padding_idx).to(idx.device), self.padding_idx)
        position_embeddings = self.pos_emb(positions)
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits_eval = logits[:,-y.shape[1]:,:].transpose(-1,-2)
        loss = F.cross_entropy(logits_eval, y)
        logs = {"loss": loss}
        # return {"loss": loss, "log": logs}
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        logits_eval = logits[:,-y.shape[1]:,:].transpose(-1,-2)
        loss = F.cross_entropy(logits_eval, y)
        self.log('val_loss', loss)
        return loss

    # def validation_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.log('val_loss', avg_loss)

    def test_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        logits_eval = logits[:,-y.shape[1]:,:].transpose(-1,-2)
        loss = F.cross_entropy(logits_eval, y)
        self.log('test_loss', loss)
        return loss

    # def test_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        # self.log('test_loss', avg_loss)


