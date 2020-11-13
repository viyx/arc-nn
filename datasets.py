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


# class ColorPermutation():
#     # x, y, x_test, y_test = data

#     def __init__(self, max_permutations=1000, max_colors=10, data=None):
#         self.max_permutations = max_permutations
#         self.max_colors = max_colors
#         self.name = 'color'

#         if(data is not None):
#             u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
#             colors_to_idx = np.zeros(self.max_colors, dtype=np.long)
#             colors_to_idx[list(u_colors)] = range(len(u_colors))
#             self.norms = [[colors_to_idx[ep] for ep in d] for d in data]
#             perms = np.array(list(itertools.permutations(range(self.max_colors), len(u_colors))))
#             ind = np.random.choice(len(perms), size=min(self.max_permutations, len(perms)), replace=False)
#             self.permutations = perms[ind]

#     @classmethod 
#     def from_data(cls, max_permutations, max_colors, data):
#         return ColorPermutation(max_permutations, max_colors, data)

#     def __len__(self):
#         return len(self.permutations)

#     def __getitem__(self, id):
#         permutation = self.permutations[id]
#         permuted_data = tuple([[permutation[i] for i in d] for d in self.norms])
#         return permuted_data




from math import factorial
import functools
from itertools import permutations
class LightWeightColorPermutation():
    # TODO make shuffling for permutation when limit exists
    # TODO now we take first #limit permutations, it shifts color distribution to first colors (0,1,2,3..)
    def __init__(self, limit=1000, max_colors=10):
        self.limit = limit
        self.max_colors = max_colors

    @functools.lru_cache(10)
    def P(self, n, r):
        "Return permutation counts. Order matters."
        return factorial(n) // factorial(n - r)

    def count(self, data):
        """Number of permutations per one task.
        Find #unique colors in task and then count all permutations. P(n,r).
        Method should be fast."""

        u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
        c = self.P(self.max_colors, self.max_colors - len(u_colors))
        return min(self.limit, c)

    # @classmethod
    # def _get_permutation(cls, n, r, idx):
    #     "Fast equivalent of `list(itertools.permutations(range(n), r))[id]`"
    #     digits = []

    #     for i in range(r):
    #         c = cls.P(n - i, r - i)
    #         q, _ = divmod(c, n - i)
    #         q1, r1 = divmod(idx, q)
    #         shift = len(list(filter(lambda x: x <= q1, digits)))
    #         digits.append(q1 + shift)
    #         idx = r1

    #     return tuple(digits)
    #     # return int(''.join(map(str, digits)))

    @functools.lru_cache(10)
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

        # replace all data
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
        "Return transformed task only if `id` is `int` and transforms exist"

        # id to task name
        if(type(id) is int and not self.transforms):
            id = self.tasks[id]

        if(type(id) is not int): # id is str, untransformed task
            with open(id) as f:
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

                # (list(np.array(ndim=2)), ...)
                return train_x, train_y, test_x, test_y
        else:
            # make all permuations
            task_idx = np.searchsorted(self.breaks, id) - 1
            task = self.tasks[task_idx] # str
            break_len = self.breaks[task_idx + 1] - self.breaks[task_idx]
            original_data = self[task]
            per_data = original_data

            # get index for each transform and transform data
            for t in self.transforms:
                c = t.count(original_data)
                q, r = divmod(break_len, c)
                per_data = t.transform_data(q, per_data)
                break_len = r

            return per_data



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
        # max flatten y size with special tokens(30 end_lines,1 end_episode)
        self.max_y_size = 30 * 30 + 30 + 1
        
        # make special tokens 
        self.end_line_token = n_colors + 0    # array of shape (10, 3) has 10 end_lines
        self.promt_token = n_colors + 1       # promt after every x
        self.end_episode_token = n_colors + 2 # end of episode
        self.pad_token = n_colors + 3       # padding token (expand to n_context from the begining)
        
        self.vocab_size = n_colors + 4
        
    def __len__(self):
        return len(self.ds)
    
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
        xy = list(map(self.flat_pair, zip(x, y)))
        xy = np.concatenate(xy, axis=None)
        
        #flat test pair
        
        # take only the first test episode (may be >1)
        # if we keep all test episodes batching would be harder to control
#         if(len(x_test) > 1):
        x_test = x_test[0]
        y_test = y_test[0]
            
        xt = self.flat_2D_field(x_test)
        yt = self.flat_2D_field(y_test)
        
        # just add end of episode
        y = np.concatenate([yt, self.end_episode_token], axis=None)
        
        # pad y to max flattened 2D field
        if(len(y) < self.max_y_size and self.padding):
            y = self.pad(y, self.max_y_size, 'right')
        
        # context: concat all
        x = np.concatenate([xy, xt, self.promt_token, y], axis=None)
        
        # padding
        if(len(x) < self.n_context and self.padding): # expand sample to n_context
            x = self.pad(x, self.n_context, 'left')
            
        return x, y
    
    def __getitem__(self, id):
        "Get raw example, then flat it and add special symbols."
        x, y, x_test, y_test = self.ds[id]
        x, y = self.flat_all_sample(x, y, x_test, y_test)
        return x, y


from download import try_download_from_bucket
import pickle
import logging
md_logger = logging.getLogger('MedianDataset')


class MedianDataset():
    # TODO do abstract class
    # TODO api for building datasets
    def __init__(self, download=True, datadir='./data/datasets/median/', files = ['train.pickle', 'test.pickle', 'val.pickle']):
        from pathlib import Path
        Path(datadir).mkdir(parents=True, exist_ok=True)
        self.datadir = datadir
        self.files = files
        if download:
            try_download_from_bucket(datadir, files)
            self.datasets = []
            for f in files:
                with open(datadir + f, 'rb') as file:
                    self.datasets.append(pickle.load(file))
        else: self.create_new()

    def create_new(self):
        "Find tasks with median x length. Create train, test, val ARCDatasets."

        # TODO remove hardcode
        n_context = 2048
        ds = LightARCDataset()
        gpt_ds = GPTDataset(ds, 10, n_context, True)
        lxs = []

        for id in range(len(ds)):
            x_gpt, _ =  gpt_ds[id]
            lxs.append(len(x_gpt))
            
            # tests
            #check special token counts
            x, *_ = ds[id]
            assert (x_gpt == gpt_ds.end_episode_token).sum() == len(x) + 1, "End of episodes missmatched."
            assert (x_gpt == gpt_ds.promt_token).sum() == len(x) + 1, "Promts missmatched."
        #     assert (x_gpt == no_aug_ds.new_line).sum() == len(x) + 1, "Promts missmatched."
            
        lxs = pd.Series(lxs)
        md_logger.info('Median length : {}'.format(lxs.median()))
        indices = lxs[lxs <= n_context].index.tolist()
        median_tasks = np.array(ds.tasks)[indices].tolist()

        train, test, val = .8, .1, .1
        train = int(train * len(median_tasks))
        test = int(test * len(median_tasks))
        train, test, val = np.array_split(median_tasks, [train, test + train])

        md_logger.info('Lengths before transforms. train : {}, test : {}, val : {}'. format(len(train), len(test), len(val)))

        train = LightARCDataset(tasks=train, transforms=[LightWeightColorPermutation(max_colors=10, limit=10000)])
        test = LightARCDataset(tasks=test)
        val = LightARCDataset(tasks=val, transforms=[LightWeightColorPermutation(max_colors=10, limit=100)])

        md_logger.info('Lengths after transforms. train : {}, test : {}, val : {}'. format(len(train), len(test), len(val)))
        self.datasets = [train, test, val]

        for i, file in enumerate(self.files):
            with open(self.datadir + file, 'wb') as f:
                # f.writelines(['123', '321'])
                pickle.dump(self.datasets[i], f)

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

