import glob
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import itertools
import math
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from copy import deepcopy


class ColorPermutation():
    # x, y, x_test, y_test = data

    def __init__(self, max_permutations=1000, max_colors=10, data=None):
        self.max_permutations = max_permutations
        self.max_colors = max_colors
        self.name = 'color'

        if(data is not None):
            u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
            colors_to_idx = np.zeros(self.max_colors, dtype=np.long)
            colors_to_idx[list(u_colors)] = range(len(u_colors))
            self.norms = [[colors_to_idx[ep] for ep in d] for d in data]
            perms = np.array(list(itertools.permutations(range(self.max_colors), len(u_colors))))
            ind = np.random.choice(len(perms), size=min(self.max_permutations, len(perms)), replace=False)
            self.permutations = perms[ind]

    @classmethod 
    def from_data(cls, max_permutations, max_colors, data):
        return ColorPermutation(max_permutations, max_colors, data)

    def __len__(self):
        return len(self.permutations)

    def __getitem__(self, id):
        permutation = self.permutations[id]
        permuted_data = tuple([[permutation[i] for i in d] for d in self.norms])
        return permuted_data


def aug_task(task, arc_ds, max_permutations, max_colors):
    data = arc_ds[task]
    inst = ColorPermutation.from_data(max_permutations, max_colors, data)
    return inst


class ARCDataset:
    def __init__(self, tasks=None, augs=None, data_folder='./data'):
        if(tasks is None):
            train_tasks = glob.glob(data_folder + '/training/*.json')
            eval_tasks = glob.glob(data_folder + '/evaluation/*.json')
            self.tasks = train_tasks + eval_tasks
        else:
            assert len(tasks) >= 1, 'Specify at least one task.'
            self.tasks = tasks

        #work only with ColorPermutation
        self.aug_tasks = None
        if(augs is not None):
            for aug in augs:
                f = partial(aug_task, arc_ds=self, max_permutations=aug.max_permutations, max_colors=aug.max_colors)
                mcp = min(mp.cpu_count(), len(tasks)) #aware case when tasks < cpu
                with Pool(mcp) as p:
                    insts = p.map(f, self.tasks, len(self.tasks) // mcp)

            breaks = [0] + list(map(len, insts))
            breaks = np.cumsum(breaks)
            aug_tasks = pd.DataFrame(index=pd.IntervalIndex.from_breaks(breaks, closed='left', name='intervals'), data=insts)
            self.aug_tasks = aug_tasks.set_index([self.tasks], append=True)
                
    def __repr__(self):
        n_train = len([t for t in self.tasks if 'train' in t])
        n_eval = len([t for t in self.tasks if 'evaluation' in t])
        n_test = len([t for t in self.tasks if 'test' in t])

        return ('data = {}, train = {}, eval = {}, test = {}'
                .format(len(self.tasks), n_train, n_eval, n_test))

    def __len__(self):
        if self.aug_tasks is None:
            return len(self.tasks)
        else:
            return self.aug_tasks.index[-1][0].right

    def stoi(self, taskname):
        "Task name to index."
        return self.tasks.index(taskname)

    def itos(self, id):
        'Index to task name'
        if(self.aug_tasks is not None):
            return self.aug_tasks.loc[id].index[0]
        else:
            return self.tasks[id]

    def __getitem__(self, id):
        
        if(type(id) is str):
            id = self.stoi(id)

        if(self.aug_tasks is None):
            with open(self.tasks[id]) as f:
                sample = json.load(f)

            # lists of 2-D arrays
            train_x = list(
                map(lambda d: np.array(d['input']),  sample['train']))
            train_y = list(
                map(lambda d: np.array(d['output']), sample['train']))
            test_x = list(
                map(lambda d: np.array(d['input']),  sample['test']))

            #submission case
            test_y = None

            #not submission case
            if('output' in sample['test'][0]):
                test_y = list(map(lambda d: np.array(d['output']), sample['test']))

            return train_x, train_y, test_x, test_y
        else:
            v = self.aug_tasks.loc[(id, slice(None)),][0]
            aug_index = id - v.index[0][0].left
            aug_inst = v.values[0]
            return aug_inst[aug_index]


    # def filter_tasks(self, filters):
    #     "Make filter by dataset boolean features."

    #     if(not isinstance(filters, (list, tuple))):
    #         filters = [filters]
    #     mask = (self.features[filters] == 1).all(axis=1)
    #     tasks = self.features.index[mask].tolist()
    #     return tasks

    # def _add_features(self):
    #     features_list = \
    #         ['eq_shape_io', #bool
    #          'eq_shape_i',  #bool
    #          'eq_color_i',  #bool
    #          'eq_color_o']  #bool
    #     data = []

    #     for t in self.tasks:
    #         x, y, *_ = self[t]
    #         pairs = zip(x, y)

    #         # eq_shape_io
    #         # the same shapes of inputs and outputs by episode
    #         eq_shape_io = \
    #             all([np.shape(s[0]) == np.shape(s[1]) for s in pairs])

    #         # eq_shape_i
    #         # the same shapes of inputs by demo
    #         shapes = list(map(np.shape, x))
    #         eq_shape_i = len(set(shapes)) == 1

    #         # eq_color_i
    #         # the same colors of inputs by demo
    #         colors = list(map(tuple, map(np.unique, x)))
    #         eq_color_i = len(set(colors)) == 1

    #         # eq_color_o
    #         # the same colors of outputs by demo
    #         colors = list(map(tuple, map(np.unique, y)))
    #         eq_color_o = len(set(colors)) == 1

    #         data.append((eq_shape_io, eq_shape_i, eq_color_i, eq_color_o))

    #     df = pd.DataFrame(index=self.tasks, columns=features_list, data=data)
    #     return df


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
        self.target_size = 30*30+30+1 # max flatten y size with special tokens(end_lines, end_episode)
        
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
            "Pad sequence to left or right."
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
        # max_tokens_on_field = 30**2 + 30 + 1
        if(len(y) < self.target_size and self.padding):
            y = pad(y, self.target_size, 'right')
        
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


import os
import requests
import logging


BUCKET_ADDR = 'https://storage.googleapis.com/viy_data/'

logger = logging.getLogger(__name__)

def try_download_from_bucket(datadir, files):
    "Download files from google cloud if not exist."
    for filename in files:
        file = os.path.join(datadir, filename)
        os.makedirs(datadir, exist_ok=True)
        if not os.path.exists(file):
            logger.info(f'start download: `{file}`')
            url = BUCKET_ADDR + file
            r = requests.get(url)
            with open(file, 'wb') as desc:
                desc.write(r.content)



import pickle


class MedianDataset():
    def __init__(self, datadir='data/datasets/median/'):
        files = ['test_dataset.pickle', 'train_dataset.pickle']
        try_download_from_bucket(datadir, files)
        with open(datadir + files[1], 'rb') as f:
            self.train_dataset = pickle.load(f)
        with open(datadir + files[1], 'rb') as f:
            self.test_dataset = pickle.load(f)

    def get_test(self):
        return self.test_dataset

    def get_train(self):
        return self.train_dataset

###
#Generation
###


# import numpy as np

# class X:
# #     ds_train_params = dict(
# #     n_episodes=1, #number of episodes
# #     colors=2,
# #     fig_w=2, #figure width(0 axis)
# #     fig_h=2, #figure height(1 axis)
# #     field_w=10,  #episode field width
# #     field_h=10,   #episode field height
# #     n_figs_on_field=3  #how much figures exepect to see on the field(2 figs can join in one)
# # )
#     def __init__(self, length_limit, **kwargs):
#         self.length_limit = length_limit
    
#         for k,v in kwargs.items():
#                 setattr(self, k, v)

#         #approx n unique samples in dataset
#         self.length_max = (self.colors ** (self.fig_w * self.fig_w) * self.field_w * self.field_h)\
#             ** self.n_figs_on_field

#         assert length_limit < length

#         #create empty field
#         x = np.zeros((length_limit, self.n_episodes, self.field_h, self.field_w), np.long)

#         #generate figures
#         figs = np.random.randint(0, self.colors,
#                                 (length_limit, self.n_episodes, self.n_figs_on_field,
#                                     self.fig_h, self.fig_w), np.int8)

#         #create coordinates for future figures
#         fig_x = np.random.randint(0, self.field_h - self.fig_h, (length_limit, self.n_episodes,
#                             self.n_figs_on_field), np.int8)
#         fig_y = np.random.randint(0, self.field_w - self.fig_w, (length_limit, self.n_episodes,
#                             self.n_figs_on_field), np.int8)
        
#         #paste figures in fields
#         ind = np.indices((length_limit, self.n_episodes), np.intp)
#         for i in range(self.n_figs_on_field):
#             for _fig_x in range(self.fig_h):
#                 for _fig_y in range(self.fig_w):
#                     x[ind[0], ind[1], fig_x[..., i] + _fig_x, fig_y[..., i] + _fig_y] =\
#                         figs[ind[0], ind[1], i, _fig_x, _fig_y]
#         x = _remove_empty(x)
#         self.x = x

#     def _remove_empty(self):
#         #remove samples with at least one empty episode
#         m = (x.sum((-1,-2)) > 0).all(1)
#         return = x[m]

#     @property
#     def length_max(self):
#         return self.length_max


# # class MoveTransform:
# #     def __init__(self, x):
# #         n, n_ep, field_h, field_w = x.shape

# #         #create arrays with shift values like [-3,-2,-1,1,2,3]

# #         #minus - shift left, plus - shift right, zero - no shift
# #         shifts_x_ind = np.arange(-field_h + 1, field_h)

# #         #minus - shift up, plus - shift down, zero - no shift
# #         shifts_y_ind = np.arange(-field_w + 1, field_w)

# #         #remove zero shift
# #         shifts_x_ind = shifts_x_ind[shifts_x_ind != 0]
# #         shifts_y_ind = shifts_y_ind[shifts_y_ind != 0]

# #         #make shift matrices
# #         shifts_x_mat = np.array([np.eye(field_h, field_h, i) for i in shifts_x_ind)])
# #         shifts_y_mat = np.array([np.eye(field_w, field_w, i) for i in shifts_y_ind)])

# #         # trans = np.flip(np.eye(field_h, field_w), 1)
# #         # no_trans = np.eye(field_h, field_w)

# #         #sample shifts
# #         x_shift_sample = np.random.normal(0, field_h // 3, n).astype(int).clip(-field_h + 1, field_h - 1)
# #         y_shift_sample = np.random.normal(0, field_w // 3, n).astype(int).clip(-field_w + 1, field_w - 1)

# #         shifts_x_mat[x_shift_sample]


    #     return 'print'

