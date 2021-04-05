from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

from torch.utils.data import Dataset
from omegaconf import DictConfig

from .transforms import Transform


class AbstractDataset(ABC):
    "High-level dataset."
    train_test_val_transforms = \
        Tuple[Optional[List[Transform]],
              Optional[List[Transform]],
              Optional[List[Transform]]]

    def __init__(self, config: DictConfig,
                 transforms: train_test_val_transforms) -> None:
        self.split = eval(config.split)
        assert sum(self.split) == 1
        self.config = config
        self.transforms = transforms

    @abstractmethod
    def _create(self) -> Tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError

    @property
    def train_test_val(self) -> Tuple[Dataset, Dataset, Dataset]:
        if(getattr(self, '_datasets', None) is None):
            self._datasets = self._create()
        return self._datasets

# class AbstractDataset(ABC):
#     "High-level dataset."
#     def __init__(self,
#         config,
#         # files=['train.pickle', 'test.pickle', 'val.pickle'],
#         ):

#         self.split = eval(config.split)
#         assert sum(self.split) == 1
#         self.config = config
#         # Path(config.datadir).mkdir(parents=True, exist_ok=True)
#         # self.datadir = config.datadir
#         # self.files = files
#         self.datasets = []
#         self._create()
#         # # if config.download:
#         # #     #TODO add downloading and saving to cloud
#         # #     for f in files:
#         # #         with open(self.datadir + f, 'rb') as file:
#         # #             self.datasets.append(pickle.load(file))
#         # else: self._create_new_dataset()

#     @abstractmethod
#     def _create(self):
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def train_test_val(self) -> list(Dataset):
#         return self.datasets



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

