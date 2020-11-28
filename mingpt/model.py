import math
import logging
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


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
        self.register_buffer("mask_tril", torch.tril(torch.ones(config.n_context, config.n_context))
                                     .view(1, 1, config.n_context, config.n_context))
        # tcontext mask to ensure we dont learn predict context itself
        # self.register_buffer("mask_rect", torch.cat([torch.zeros((config.n_context - config.target_length, config.n_context)),
                                    # torch.ones((config.target_length, config.n_context))])
                                    #  .view(1, 1, config.n_context, config.n_context))
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
        # att = att.masked_fill(self.mask_rect[:,:,:T,:T] == 0, 0.0) #hide perm context from backward pass
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

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.pad_token = config.pad_token
        self.promt_token = config.promt_token
        self.end_episode_token = config.end_episode_token
        self.block_size = config.n_context
        self.target_size = config.target_length


        # positions start from `1` as `0` token reserved as padding index for positions
        self.register_buffer('arange', torch.arange(1, self.target_size+1))
        # self.register_buffer('pos', torch.zeros(self.block_size))
        # self.register_buffer('posAB', torch.zeros(self.block_size))

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(self.block_size + 1, config.n_embd, padding_idx=0)
        self.pos_emb_ab = nn.Embedding(2 + 1, config.n_embd, padding_idx=0)
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size * 3, "Cannot forward, model block size is exhausted."

        idx = idx.view(b,3,-1)
        # forward the GPT model
        # position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector


        # for token in 
        # positions = self.positions.masked_fill((idx == self.pad_token).to(idx.device), 0)
        
        # first_not_pad_idx = torch.nonzero(idx != self.pad_token)[0]

        # len(end_eps) - len(promts) = 1
        # promts = torch.nonzero(idx != self.promt_token)
        # end_eps = torch.cat([first_not_pad_idx, torch.nonzero(idx != self.end_episode_token)])

        # for i in range(len(promts)):
        #     self.pos[end_eps[i]:promts[i]+1] = self.arange[promts[i]-end_eps[i]+1:]
        #     self.posAB[end_eps[i]:promts[i]+1] = 1 # x
        #     self.pos[promts[i]:end_eps[i+1]+1] = self.arange[end_eps[i+1]-promts[i]+1:]
        #     self.posAB[promts[i]:end_eps[i+1]+1] = 2 # y

        #set `0` for positions with padding index
        # positions = self.arange.masked_fill((idx == self.padding_idx).to(idx.device), 0)
        token_embeddings = self.tok_emb(idx[:,0,:]) # each index maps to a (learnable) vector
        pos_embeddings = self.pos_emb(idx[:,1,:])
        pos_ab_embeddings = self.pos_emb_ab(idx[:,2,:])
        x = self.drop(token_embeddings + pos_embeddings + pos_ab_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            logits_eval = logits[:,-targets.shape[1]:,:].transpose(-1,-2) # take only last logits as we need to predict them from context
            loss = F.cross_entropy(logits_eval, targets, ignore_index=self.pad_token)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add('pos_emb')
        # no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        betas = eval(train_config.betas)
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=betas)
        return optimizer


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--n_embd", type=int, default=8)
        parser.add_argument("--n_head", type=int, default=2)
        parser.add_argument("--n_layer", type=int, default=2)
        parser.add_argument("--embd_pdrop", type=float, default=0.0)
        parser.add_argument("--attn_pdrop", type=float, default=0.1)
        parser.add_argument("--resid_pdrop", type=float, default=0.1)
        parser.add_argument("--vocab_size", type=int, default=14)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=0.1)
        parser.add_argument("--momentum", type=float, default=0.5)
        parser.add_argument("--betas", type=str, default='(0.9, 0.95)')
        return parser