import glob
import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import itertools
import math
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from copy import deepcopy
from math import factorial
from itertools import permutations


class Transform:
    "Abstract class which all transformation classes should derive."
    def __init__(self, limit):
        self.limit = limit

    def count(self, data):
        "Return number of all possible transformations for single task. Can't exceed the limit."
        raise(NotImplementedError())

    def transform_data(self, idx, data):
        """Return one transformation from all possible for data.
        `idx` should be less `min(self.limit, self.count(data))`.
        """
        raise(NotImplementedError())
    

class LightWeightColorPermutation(Transform):
    # TODO make shuffling for permutation when limit exists
    # TODO now we take first #limit permutations, it shifts color distribution to first colors (0,1,2,3..)
    def __init__(self, limit=1000, max_colors=10):
        super().__init__(limit)
        self.max_colors = max_colors

    def P(self, n, r):
        "Return permutation counts. Order matters."
        return factorial(n) // factorial(n - r)

    def count(self, data):
        """Number of permutations per one task.
        Find #unique colors in task and then count all permutations. P(n,r).
        """

        u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
        c = self.P(self.max_colors, len(u_colors))
        return min(self.limit, c)

    def _get_permutations(self, n, r):
        return list(permutations(range(n), r))

    def transform_data(self, idx, data):
        """Get permutated data.
        Find unique colors in data, then estimate permutation for them
        and replace data with the permutation.

        """
        assert idx <= self.limit, "Id should be less or equal then limit. Check caller function."
        u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
        u_colors = np.array(list(u_colors))

        # tuple like (1,5,6)
        p = self._get_permutations(self.max_colors, len(u_colors))[idx]

        # replace unique colors with permuted ones
        m = np.zeros(self.max_colors, dtype=np.long)
        m[u_colors] = p

        # replace color in all data
        p_data = tuple([[m[ep] for ep in d] for d in data])
        return p_data


class LightARCDataset:
    def __init__(self, tasks=None, transforms=None, data_folder='./data'):
        if(tasks is None): # load all tasks
            train_tasks = glob.glob(data_folder + '/training/*.json')
            eval_tasks = glob.glob(data_folder + '/evaluation/*.json')
            self.tasks = train_tasks + eval_tasks
        else:
            assert len(tasks) >= 1, 'Specify at least one task.'
            self.tasks = tasks

        # populate breaks for futher navigation througout transforms
        # each transform has `count()` method which returns #n new tasks
        # we record #n transformations per task in `breaks` which contain intervals
        if transforms:
            breaks = [0]
            for task in self.tasks:
                per_task_cnt = 0
                for tr in transforms:
                    per_task_cnt += tr.count(self[task])
                breaks.append(per_task_cnt)
            
            # `breaks` will contain breaks for all dataset.
            # for example, (break[1] - break[0]) shows transformation count for first task
            self.breaks = np.cumsum(breaks)
            assert len(self.breaks) == len(self.tasks) + 1

        self.transforms = transforms

    def __repr__(self):
        n_train = len([t for t in self.tasks if 'train' in t])
        n_eval = len([t for t in self.tasks if 'evaluation' in t])
        n_test = len([t for t in self.tasks if 'test' in t])
        n_trasformed = len(self)

        return ('ARC folder distribution: all = {}, train = {}, eval = {}, test = {}. Transformed : {}'
                .format(len(self.tasks), n_train, n_eval, n_test, n_trasformed))

    def __len__(self):
        if self.transforms:
            return self.breaks[-1]
        else:
            return len(self.tasks)

    def __getitem__(self, id):
        "Find task and make tranformations."

        # convert id to taskname if no transorms
        if(type(id) is int and self.transforms is None):
            id = self.tasks[id]

        if(type(id) is not int): # id is str, untransformed task
            with open(id) as f:
                sample = json.load(f)
            # map json to lists of 2-D arrays
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

                # list(np.array(ndim=2)), list(np.array(ndim=2)) ...
                return train_x, train_y, test_x, test_y
        else:
            # find original task for `id` and make all transforms
            original_task_idx = np.searchsorted(self.breaks, id)
            original_taskname = self.tasks[original_task_idx] # str
            original_data = self[original_taskname]
            
            permuted_data = original_data
            id -= self.breaks[original_task_idx]
            # get index for each transform and do transform
            for t in self.transforms:
                c = t.count(original_data)
                q, r = divmod(id, c)
                permuted_data = t.transform_data(q, permuted_data)
                id = r

            return permuted_data

    def plot(self, id):
        import matplotlib.pyplot as plt
        from matplotlib import colors

        def plot_one(ax, data, train_or_test, input_or_output):
            cmap = colors.ListedColormap(
                ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
                '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
            norm = colors.Normalize(vmin=0, vmax=9)
            
            ax.imshow(data, cmap=cmap, norm=norm)
            ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
            ax.set_yticks([x-0.5 for x in range(1+len(data))])
            ax.set_xticks([x-0.5 for x in range(1+len(data[0]))])     
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(train_or_test + ' '+input_or_output)

        def plot_task(task):
            """
            Plots the first train and test pairs of a specified task,
            using same color scheme as the ARC app
            """    
            x_train, y_train, x_test, y_test = task
            num_train = len(x_train)
            _, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
            for i in range(num_train):     
                plot_one(axs[0,i], x_train[i],'train','input')
                plot_one(axs[1,i], y_train[i],'train','output')        
            plt.tight_layout()
            plt.show()        
                
            num_test = len(x_test)
            _, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2), squeeze=False)
            
            for i in range(num_test):
                plot_one(axs[0,i], x_test[i],'test','input')
                plot_one(axs[1,i], y_test[i],'test','output')  
            plt.tight_layout()
            plt.show() 
        
        item = self[id]
        plot_task(item)
        


class GPTDataset(Dataset):
    """Flat 2D samples and add specials tokens.
    
    General scheme:
    
    flatten(x) + `promt` + flatten(y) + `end_episode`
    
    Here `flatten` is:
    flat 2D array and add `end_line` in the end of every line.
    """
    def __init__(self, dataset, n_colors, n_context, target_length, padding=False):
        self.dataset = dataset
        self.n_colors = n_colors
        self.n_context = n_context
        self.padding = padding # expand x to n_context
        # max flatten y size with special tokens(30 end_lines,1 end_episode)
        self.target_length = target_length
        
        # make special tokens 
        self.end_line_token = n_colors + 0    # array of shape (10, 3) has 10 end_lines
        self.promt_token = n_colors + 1       # promt after every x
        self.end_episode_token = n_colors + 2 # end of episode
        self.pad_token = n_colors + 3       # padding token (expand to n_context from the begining)
        
        self.vocab_size = n_colors + 4
        
    def __len__(self):
        return len(self.dataset)
    
    def flat_2D_field(self, x):
        "Add column of `end_line` tokens and flat 2D array."
        a = np.array([[self.end_line_token]] * x.shape[0], dtype=np.long)
        a = np.hstack([x, a])
        return a.ravel()

    def flat_pair(self, pair):
        "Flat x, y pairs and add `promt` and `end_episode` tokens"
        x, y = pair
        x = self.flat_2D_field(x)
        y = self.flat_2D_field(y)
        return np.concatenate([x, self.promt_token, y, self.end_episode_token], axis=None)
    
    def pad(self, seq, to, direct):
        "Pad sequence to left or right."
        x = np.array([self.pad_token] * to, dtype=np.long)
        if(direct == 'left'):
            x[-len(seq):] = seq
        if(direct == 'right'):
            x[:len(seq)] = seq
        return x

    def flat_all_sample(self, x, y, x_test, y_test):
        # flat train pairs
        xy_train = list(map(self.flat_pair, zip(x, y)))
        xy_train = np.concatenate(xy_train, axis=None)
        
        #flat test pair
        
        # take only the first test episode (may be >1)
        # if we keep all test episodes batching would be harder to control
#         if(len(x_test) > 1):
        x_test = x_test[0]
        y_test = y_test[0]
            
        x_test = self.flat_2D_field(x_test)
        y_test = self.flat_2D_field(y_test)
        
        # just add end of episode
        y = np.concatenate([y_test, self.end_episode_token], axis=None)
        
        # pad y to max flattened 2D field
        if(len(y) < self.target_length and self.padding):
            y = self.pad(y, self.target_length, 'right')
        
        # context: concat all
        x = np.concatenate([xy_train, x_test, self.promt_token, y[:-1]], axis=None)
        
        # padding
        if(len(x) < self.n_context and self.padding): # expand sample to n_context
            x = self.pad(x, self.n_context, 'left')
            
        # we dont make shift like x = data[:-1], y = data[1:] to decrease memory flow
        # we will cut predictions to max_target_size in model in order to calculate criterion
        # len(x) = n_context, len(y) = max_target_size
        return x, y
    
    def __getitem__(self, id):
        "Get raw example, then flat it and add special symbols."
        x, y, x_test, y_test = self.dataset[id]
        x, y = self.flat_all_sample(x, y, x_test, y_test)
        return x, y


###
### Here you can find different high-level configuraitons of datasets
### 


from download import try_load_and_save_from_bucket_if_not_exist
import pickle
import logging
from pathlib import Path
from argparse import ArgumentParser


class AbstractDataset():
    "Common object for all datasets."
    def __init__(
        self,
        datadir,
        download=True,
        files=['train.pickle', 'test.pickle', 'val.pickle'],
        transforms=[None, None, None],  # train, test, val transformations
        split = '(0.8,0.1,0.1)',
        *args,
        **kwargs):

        self.split = eval(split)
        assert sum(self.split) == 1
        Path(datadir).mkdir(parents=True, exist_ok=True)
        self.datadir = datadir
        self.files = files
        self.transforms = transforms
        self.datasets = []

        if download:
            # TODO extract check files logic from next-line func
            try_load_and_save_from_bucket_if_not_exist(datadir, files)
            for f in files:
                with open(datadir + f, 'rb') as file:
                    self.datasets.append(pickle.load(file))
        # TODO save after create
        else: self.create_new_dataset(**kwargs)

    def create_new_dataset(self, **kwargs):
        raise(NotImplementedError())

    @classmethod
    def add_data_specific_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--split", type=str, default='(0.8,0.1,0.1)')
        parser.add_argument("--download", action="store_true")
        # parser.add_argument("--datadir", type=str, default='./data/datasets/default/')
        return parser


class MaxNDataset(AbstractDataset):
    "Fitler tasks by length of context."
    # TODO api for building datasets
    def __init__(
        self,
        target_length,
        n_colors=10,
        maxn=2048, # median lenght is ~2048
        # datadir='data/datasets/maxn/',
        *args,
        **kwargs):

        self.logger = logging.getLogger('MaxNDataset')
        self.maxn = maxn
        self.target_length = target_length
        self.n_colors = n_colors
        super().__init__(*args, **kwargs)

    def create_new_dataset(self, **kwargs):
        "Find tasks with length <= maxn. Create train, test, val datasets."

        # find tasks with length <= maxn
        ds = LightARCDataset()
        gpt_ds = GPTDataset(ds, self.n_colors, self.maxn, self.target_length, True)
        lxs = []

        for id in range(len(ds)):
            x_gpt, _ =  gpt_ds[id]
            lxs.append(len(x_gpt))
            
            # tests
            #check special token counts
            # x, *_ = ds[id]
            # assert (x_gpt == gpt_ds.end_episode_token).sum() == len(x) + 1, "End of episodes missmatched."
            # assert (x_gpt == gpt_ds.promt_token).sum() == len(x) + 1, "Promts missmatched."
        #     assert (x_gpt == no_aug_ds.new_line).sum() == len(x) + 1, "Promts missmatched."
            
        lxs = pd.Series(lxs)
        self.logger.info('Median length : {}'.format(lxs.median()))
        indices = lxs[lxs <= self.maxn].index.tolist()
        maxn_tasks = np.array(ds.tasks)[indices].tolist()

        # split tasks
        train, test, val = self.split
        train = int(train * len(maxn_tasks))
        test = int(test * len(maxn_tasks))
        train, test, val = np.array_split(maxn_tasks, [train, test + train])
        self.logger.info('Lengths before transforms. train : {}, test : {}, val : {}'.
            format(*map(len, [train, test, val])))

        # make datasets
        tasks = [train, test, val]
        for tasks, transform, file in zip(tasks, self.transforms, self.files):
            arc = LightARCDataset(tasks=tasks, transforms=transform)
            gpt = GPTDataset(arc, self.n_colors, self.maxn, self.target_length, True)
            with open(os.path.join(self.datadir, file), 'wb') as f:
                pickle.dump(gpt, f)
            self.datasets.append(gpt)

        self.logger.info('Lengths after transforms. train : {}, test : {}, val : {}'.
            format(*map(len, self.datasets)))

    @classmethod
    def add_data_specific_args(cls, parent_parser):
        parent_parser = super().add_data_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--target_length", type=int, default=30*30+30+1)
        parser.add_argument("--maxn", type=int, default=2048)
        parser.add_argument("--n_colors", type=int, default=10)
        parser.add_argument("--datadir", type=str, default='data/datasets/maxn/')
        return parser

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

