import numpy as np
import logging
from typing import List, Tuple, Optional, TypeVar, Any, Generic

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from omegaconf import DictConfig

from .base import AbstractDataset
from .arc import UnfoldARCDataset
from .transforms import Transform


TRAIN_TEST_VAL_TRANSFORMS = \
        Tuple[Optional[List[Transform]],
              Optional[List[Transform]],
              Optional[List[Transform]]]

PAIR_T = Tuple[Any, Any]

POSITION_PAD_TOKEN = 0


# class GPTDataset(Dataset):
#     """Flat 2D samples and add specials tokens in this way:
#     flatten(x) + `promt` + flatten(y) + `end_episode`.
#     Here `flatten(arg)` is:
#     flat 2D array and add `end_line` in the end of every line.
#     """
#     def __init__(self, dataset, config):
#         self.dataset = dataset
#         self.n_colors = config.n_colors
#         self.n_context = config.n_context
#         self.padding = config.padding # expand x to n_context
#         # max flatten y size with special tokens(30 end_lines,1 end_episode)
#         self.target_length = config.target_length
#         self.add_pos_tokens = config.add_pos_tokens

        
#         # make special tokens 
#         self.end_line_token =  config.end_line_token           # array of shape (10, 3) has 10 end_lines
#         self.promt_token = config.promt_token                  # promt after every x
#         self.end_episode_token = config.end_episode_token      # end of episode
#         self.pad_token = config.pad_token                      # padding token (expand to n_context from the begining)

#         # check similarity
#         assert len(set([self.end_line_token, self.pad_token, self.promt_token, self.end_episode_token])) == 4
        
#         # add 4 special token
#         self.vocab_size = config.n_colors + 4
        
#     def __len__(self):
#         return len(self.dataset)

#     # TODO not pass if no padding needed
#     def pad(self, seq, to, direct, pad_token):
#         "Pad sequence to left or right."
#         x = np.array([pad_token] * to, dtype=np.long)
#         if(direct == 'left'):
#             x[-len(seq):] = seq
#         if(direct == 'right'):
#             x[:len(seq)] = seq
#         return x

#     def flat_all_sample(self, x, y, x_test, y_test):
#         # flat train pairs
#         tokens = [self.promt_token, self.end_episode_token, self.end_line_token]
#         xy_train = list(map(lambda xy: flat_xy(xy, *tokens), zip(x, y)))
#         xy_train = np.concatenate(xy_train, axis=None)
        
#         #flat test pair
        
#         # take only the first test episode (may be >1)
#         # if we keep all test episodes batching would be harder to control
# #         if(len(x_test) > 1):
#         x_test = x_test[0]
#         y_test = y_test[0]
            
#         x_test = flat_x(x_test, self.end_line_token)
#         y_test = flat_x(y_test, self.end_line_token)
        
#         # just add end of episode
#         y = np.concatenate([y_test, self.end_episode_token], axis=None)
        
#         # pad y to max flattened 2D field
#         if(len(y) < self.target_length and self.padding):
#             y = self.pad(y, self.target_length, 'right', self.pad_token)
        
#         # context: concat all
#         # remove last token from y
#         x = np.concatenate([xy_train, x_test, self.promt_token, y[:-1]], axis=None)
        
#         # padding
#         if(len(x) < self.n_context and self.padding): # expand sample to n_context
#             x = self.pad(x, self.n_context, 'left', self.pad_token)
            
#         # we dont make shift like x = data[:-1], y = data[1:] to decrease data flow
#         # we will cut predictions to target_length in model in order to calculate criterion
#         # len(x) = n_context, len(y) = target_length

#         # tests
#         # assert np.allclose(x[-self.target_length+1:], y[:-1])
#         # assert (x[-self.target_length+1:] != self.pad_token).sum() > 2
#         return x, y
    
#     def __getitem__(self, id):
#         "Get raw example, then flat it and add special symbols."
#         x, y, x_test, y_test = self.dataset[id]
        

#         # this ugly code adds position tokens
#         # you can see result below
#         # TODO move positinal tokens to ARCDataset
#         if(self.add_pos_tokens):
#             xy_train_pos = []
#             xy_train_pos_ab = []
#             for i in range(len(x)):
#                 x_ = ((x[i].shape[0])*(x[i].shape[1]+1))+1
#                 y_ = ((y[i].shape[0])*(y[i].shape[1]+1))+1
#                 xy_train_pos.extend(np.concatenate((np.arange(1,x_+1), np.arange(1,y_+1))))

#                 xy_train_pos_ab.extend([1]*x_)
#                 xy_train_pos_ab.extend([2]*y_)

#             x_ = ((x_test[0].shape[0])*(x_test[0].shape[1]+1))+1
#             y_ = ((y_test[0].shape[0])*(y_test[0].shape[1]+1))+1
#             y_ = np.arange(1,y_+1)

#             # pad y to max flattened 2D field
#             if(len(y_) < self.target_length and self.padding):
#                 y_ = self.pad(y_, self.target_length, 'right', 0)
#             xy_test_pos = np.concatenate((np.arange(1,x_+1), y_[:-1]))
#             xy_test_pos_ab = []
#             xy_test_pos_ab.extend([1]*x_)
#             xy_test_pos_ab.extend([2]*(len(y_)-1))

#             xy_train_pos.extend(xy_test_pos)

#             # padding
#             if(len(xy_train_pos) < self.n_context and self.padding):
#                 xy_train_pos = self.pad(xy_train_pos, self.n_context, 'left', 0)

#             xy_train_pos_ab.extend(xy_test_pos_ab)
#             # padding
#             if(len(xy_train_pos_ab) < self.n_context and self.padding):
#                 xy_train_pos_ab = self.pad(xy_train_pos_ab, self.n_context, 'left', 0)


#             # add positional tokens separately for each episode (multipling #tokens by 3)
#             # example
#             #                   |---------------------------x-----------------------|--------------------y----------------|
#             # main tokens:      [<pad>, <pad>, <pad>, <pad>, <pad>, 5 ,7 ,6, <promt>, 8, 2, 3, <end_episode>, <pad>, <pad>]
#             # pos tokens:       [0,     0,      0,      0,      0,  1 ,2 ,3,    4,    1, 2, 3,      4,           0,      0]
#             # pos_ab tokens:    [0,     0,      0,      0,      0,  1 ,1 ,1,    1,    2, 2, 2,      2,           0,      0]
            
#         x, y = self.flat_all_sample(x, y, x_test, y_test)
#         if(self.add_pos_tokens):
#             x = np.concatenate((x, xy_train_pos, xy_train_pos_ab), axis=None)

#         return x, y


# class MaxNDataset(AbstractDataset):
#     "Fitler tasks by length of context."
#     # TODO api for building datasets
#     def __init__(self, config: DictConfig,
#                  transforms: TRAIN_TEST_VAL_TRANSFORMS) -> None:
#         self.logger = logging.getLogger('MaxNDataset')
#         self.maxn = config.maxn
#         self.config = config
#         self.target_length = config.target_length
#         self.n_colors = config.n_colors
#         self.padding = config.padding
#         # self.transforms = transforms
#         # self.add_pos_tokens = config.add_pos_tokens
#         super().__init__(config=config)

#     def create_new_dataset(self):
#         "Find tasks with length <= maxn. Create train, test, val datasets."

#         # find tasks with length <= maxn
#         ds = ARCDataset(self.config)
#         gpt_ds = GPTDataset(ds, self.config)
#         lxs = []

#         for id in range(len(ds)):
#             x_gpt, _ =  gpt_ds[id]
#             lxs.append(len(x_gpt))
            
#         lxs = pd.Series(lxs) # TODO replace pandas with np
#         self.logger.info('Median length : {}'.format(lxs.median()))

#         # multiply by 3 if GPTDataset adds position tokens
#         # maxn = self.maxn if not self.add_pos_tokens else self.maxn * 3
#         indices = lxs[lxs <= self.maxn].index.tolist()
#         assert len(indices) > 0, f'No tasks with length {self.maxn}.\
#                 Check --add_pos_tokens and --padding arguments.'
#         maxn_tasks = np.array(ds.tasks)[indices].tolist()

#         # split tasks
#         train, test, val = self.split
#         train = int(train * len(maxn_tasks))
#         test = int(test * len(maxn_tasks))
#         train, test, val = np.array_split(maxn_tasks, [train, test + train])
#         self.logger.info('Lengths before transforms. train : {}, test : {}, val : {}'.
#             format(*map(len, [train, test, val])))

#         # make datasets
#         tasks = [train, test, val]
#         for tasks, transform, file in zip(tasks, self.transforms, self.files):
#             arc = ARCDataset(tasks=tasks, transforms=transform)
#             gpt = GPTDataset(arc, self.config)
#             # with open(os.path.join(self.datadir, file), 'wb') as f:
#                 # pickle.dump(gpt, f)
#             self.datasets.append(gpt)

#         self.logger.info('Lengths after transforms. train : {}, test : {}, val : {}'.
#             format(*map(len, self.datasets)))


class OneTaskOneDataset(AbstractDataset):
    def __init__(self,
                 task: str,
                 split: str,
                 transforms: TRAIN_TEST_VAL_TRANSFORMS,
                 same_train_val: bool = True) -> None:
        self.logger = logging.getLogger('OneTaskOneDataset')
        # TODO remove same_train_val
        self.same_train_val = same_train_val
        self.task = task
        super().__init__(split, transforms)
        assert len(self.split) == 2

    def _calc_train_n(self, dataset: UnfoldARCDataset) -> int:
        tr, val = self.split
        tr_num = int(tr * len(dataset))
        return tr_num

    def _create(self) -> Tuple[Dataset, Dataset, Dataset]:
        if(self.same_train_val):
            train_transform = self.transforms[0]
            uarc = UnfoldARCDataset(task=self.task,
                                    transforms=train_transform,
                                    test=False)
            n = self._calc_train_n(uarc)
            indices = np.arange(len(uarc))
            np.random.shuffle(indices)
            train_indicies = indices[:n]
            val_indicies = indices[n:]
            train_ds: Dataset = Subset(uarc, train_indicies)
            val_ds = Subset(uarc, val_indicies)

            test_transforms = self.transforms[1]
            test_ds = UnfoldARCDataset(task=self.task,
                                       transforms=test_transforms,
                                       test=True)
            return train_ds, test_ds, val_ds
        else:
            train = UnfoldARCDataset(task=self.task,
                                     transforms=self.transforms[0])
            val = UnfoldARCDataset(task=self.task,
                                   transforms=self.transforms[2])
            test = UnfoldARCDataset(task=self.task,
                                    transforms=self.transforms[1],
                                    test=True)
            return train, test, val


class OneTaskCollator:
    def __init__(self,
                 pad_token: int,
                 end_line_token: int,
                 add_pos_tokens: bool) -> None:
        self.pad_token = pad_token
        self.end_line_token = end_line_token
        self.add_pos_tokens = add_pos_tokens

    def __call__(self, batch: List[PAIR_T]) -> Tuple[Tensor, Tensor]:

        x_list, y_list = [], []

        for (x, y) in batch:
            x_list.append(flat_x(x[0], self.end_line_token))
            y_list.append(flat_x(y[0], self.end_line_token))

        # pad seq from the left
        x = pad_sequence(x_list, self.add_pos_tokens, self.pad_token)

        # pad seq from the right
        y = pad_sequence(y_list, False, self.pad_token, False)

        return x, y


def pad_sequence(sequences: List[np.ndarray],
                 add_pos_tokens: bool,
                 padding_value: int,
                 padding_left: bool = True) -> Tensor:
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Returns:
        Tensor of size ``B x T x *``
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.from_numpy(sequences[0])
    out_tensor = out_tensor.new_full(out_dims, padding_value)

    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        
        if(padding_left):
            out_tensor[i, -length:, ...] = torch.from_numpy(tensor)
        else:
            out_tensor[i, :length, ...] = torch.from_numpy(tensor)

    if(add_pos_tokens):
        with torch.no_grad():
            mask = out_tensor != padding_value
            positions = torch.cumsum(mask, 1)
            positions[~mask] = POSITION_PAD_TOKEN
            out_tensor = torch.cat((out_tensor, positions), 1)
    return out_tensor


def flat_x(x: np.ndarray, end_line_token: int) -> np.ndarray:
    "Add column of `end_line` tokens and flat 2D array."
    assert x.ndim == 2
    x_pad = np.pad(x, ((0, 0), (0, 1)), 'constant',
                   constant_values=end_line_token)
    x_ravel = x_pad.ravel()
    return x_ravel


# def add_pos_tokens(x: np.ndarray) -> np.ndarray:
#     "Create array of positions. Start from '1'"
#     l = len(x)
#     pos = np.arange(1, l)
#     return pos


def flat_xy(pair: PAIR_T,
            promt_token: int,
            end_episode_token: int,
            end_line_token: int) -> np.ndarray:
    "Flat x, y pairs and add `promt` and `end_episode` tokens."
    x, y = pair
    x = flat_x(x, end_episode_token)
    y = flat_x(y, end_episode_token)
    return np.concatenate([x, promt_token, y, end_episode_token], axis=None)



# class OneTaskOneDataset(AbstractDataset):
#     train_test_val_transforms = \
#         Tuple[Optional[List[Transform]],
#               Optional[List[Transform]],
#               Optional[List[Transform]]]

#     def __init__(self, config: DictConfig,
#                  transforms: train_test_val_transforms,
#                  split_train_val: bool = False) -> None:
#         self.logger = logging.getLogger('OneTaskOneDataset')
#         self.split_train_val = split_train_val
#         assert len(config.tasks) == 1
#         super().__init__(config, transforms)

#     class UnpackingDataset(Dataset):
#         def __init__(self, arc_dataset: ARCDataset, test: bool = False) -> None:
#             self.arc = arc_dataset
#             self.test = test

#         def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
#             if(self.test):
#                 *_, x_test, y_test = self.arc.original_dataset[0]
#                 return x_test, y_test
#             else:
#                 x_train, y_train, *_ = self.arc[idx]

#                 delim = 0
#                 for tr in self.arc.transforms:
#                     if(isinstance(tr, UnPacking)):
#                         continue
#                     delim += tr.count(self.arc.original_dataset[0])

#                 unpack_idx = len(self.arc) // delim
#                 return x_train[unpack_idx], y_train[unpack_idx]
        
#         def __len__(self) -> int:
#             if(self.test):
#                 *_, x_test, _ = self.arc.original_dataset[0]
#                 return len(x_test)
#             else:
#                 return len(self.arc)


#     def _split(self, arc_dataset: ARCDataset) -> Tuple[int, int, int]:
#         tr, test, val = self.split
#         tr = tr + test
#         tr_num = int(tr * len(arc_dataset))
#         val_num = len(arc_dataset) - tr_num
#         return tr_num, 1, val_num

#     def _create(self) -> Tuple[Dataset, Dataset, Dataset]:
#         if(self.split_train_val):
#             train_transforms = self.transforms[0]
#             arc = ARCDataset(tasks=self.config.tasks, transforms=train_transforms)
#             train_val_ds = self.UnpackingDataset(arc, test=False)
#             splits = self._split(arc)
#             indices = np.arange(len(arc))
#             np.random.shuffle(indices)
#             train_indicies = indices[:splits[0]]
#             val_indicies = indices[splits[0]:]
#             train_ds = Subset(train_val_ds, train_indicies)
#             val_ds = Subset(train_val_ds, val_indicies)
#             test_ds = self.UnpackingDataset(arc.original_dataset, test=True)
#             return train_ds, test_ds, val_ds
#         else:
#             arc_train = ARCDataset(tasks=self.config.tasks,
#                                    transforms=self.transforms[0])
#             arc_val = ARCDataset(tasks=self.config.tasks,
#                                  transforms=self.transforms[2])
#             train_ds = self.UnpackingDataset(arc_train, False)
#             test_ds = self.UnpackingDataset(
#                         arc_train.original_dataset, True)
#             val_ds = self.UnpackingDataset(arc_val, False)
#             return train_ds, test_ds, val_ds
