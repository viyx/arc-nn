import glob
import json
from typing import Tuple, List, Union, Optional, Sequence
from abc import abstractmethod
from collections.abc import Sized

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from torch.utils.data import Dataset

from .transforms import Transform


DATA_FOLDER = 'datasets/data'
T = Tuple[List[np.ndarray], ...]


class ARCBase(Dataset, Sized):
    def __init__(self, transforms: Optional[List[Transform]] = None):
        self.transforms = transforms

    @property
    def index(self) -> Sequence:
        "Contains cumulative sum over transformations per task."
        if(getattr(self, '_index', None) is None):
            if(self.transforms is None):
                self._index = np.arange(1, len(self)+1)
            else:
                self._create_index()
        return self._index

    @abstractmethod
    def _create_index(self) -> None:
        raise NotImplementedError


class ARCDataset(ARCBase):
    """Base class with access to raw data.
    It also can make and store transofmations.

    Parameters
    ----------
    tasks : List[str], default=None
        List of tasks's file names.
        If None, they will be read from the `data` folder.

    transforms : List[Transform], default=None
        List of transformations for augmentation.

    data_folder : str, default='datasets/data'
        Name of folder with raw ARC data.

    Attributes
    ----------

    original_dataset : ARCDataset
        Dataset without transformations.

    index : np.array or None
        Index for fast navigation in transformed tasks.
    """
    def __init__(self,
                 tasks: Optional[List[str]] = None,
                 transforms: Optional[List[Transform]] = None,
                 data_folder: str = DATA_FOLDER) -> None:

        super().__init__(transforms)
        if(self.transforms):
            self.original_dataset = ARCDataset(tasks=tasks,
                                               data_folder=data_folder)
        else:
            self.original_dataset = self
        self._read_tasks(data_folder, tasks)

    def _read_tasks(self, folder: str, tasks: Optional[List[str]]) -> None:
        if(tasks is None):
            train_tasks = glob.glob(folder + '/training/*.json')
            eval_tasks = glob.glob(folder + '/evaluation/*.json')
            self.tasks = train_tasks + eval_tasks
        else:
            assert len(tasks) >= 0, 'Specify at least one task.'
            self.tasks = tasks

    def _create_index(self) -> None:
        index = []
        for task in self.tasks:
            per_task_cnt = 0
            for tr in (self.transforms):
                per_task_cnt += tr.count(self.original_dataset[task])
            index.append(per_task_cnt)
        self._index = np.cumsum(index)

    def __repr__(self) -> str:
        n_train = len([t for t in self.tasks if 'train' in t])
        n_eval = len([t for t in self.tasks if 'evaluation' in t])
        n_test = len([t for t in self.tasks if 'test' in t])
        n_trasformed = len(self)

        s = 'ARC folder distribution: '
        s += 'transformed : {},'
        s += 'train = {}, '
        s += 'eval = {}, '
        s += 'test = {}'

        return s.format(n_trasformed, n_train, n_eval, n_test)

    def __len__(self) -> int:
        if self.transforms:
            return self.index[-1]
        else:
            return len(self.tasks)

    def __getitem__(self, id: Union[int, str]) -> T:
        """Find task and make tranformations.

        Parameters
        ----------

        id : {str, int}
            Name or id of task.

        Returns
        -------

        T: Tuple[List[np.ndarray], List[np.ndarray],
                 List[np.ndarray], [List[np.ndarray]]]
        Provides `x_train, y_train, x_test, y_test` np.ndarrays in one tuple.
        `y_test` can be None in kaggle submission regime.
        """
        if(not self.transforms):
            if(type(id) is int):
                filename = self.tasks[int(id)]
            else:
                filename = str(id)
            with open(filename) as f:
                sample = json.load(f)
                train_x = list(
                    map(lambda d: np.array(d['input']),  sample['train']))
                train_y = list(
                    map(lambda d: np.array(d['output']), sample['train']))
                test_x = list(
                    map(lambda d: np.array(d['input']),  sample['test']))
                # if('output' in sample['test'][-1]):
                test_y = list(
                    map(lambda d: np.array(d['output']), sample['test']))
                # else: test_y = None
                return train_x, train_y, test_x, test_y
        else:
            if(id is str):
                assert False, "Can't use taskname with tranforms."
            assert int(id) < len(self)

            # find original task
            original_task_idx = int(np.searchsorted(self.index, id, 'right'))
            original_data = self.original_dataset[original_task_idx]
            # find id for transform
            start_idx = self.index[original_task_idx-1]
            tr_id = id - start_idx
            # make transforms
            data = original_data
            for t in self.transforms:
                c = t.count(data)
                q, r = divmod(tr_id, c)
                data = t.transform_data(r, data)
                tr_id = q
            return data


class UnfoldARCDataset(Dataset):
    """Make dataset from one task.

    Parameters
    ----------
    task : str
        Task name.

    transforms : List[Transform], default=None
        List of transformations for augmentation.

    data_folder : str, default='data'
        Name of folder with raw ARC data.

    test : bool, default=False
        Whether train or test data.

    Attributes
    ----------

    original_dataset : UnfoldARCDataset
        Dataset without transformations.

    index : np.array or None
        Index for fast navigation in transformed tasks.

    x : List[np.array]
        One `x` packed in list.

    y : List[np.array]
        One `y` packed in list.
    """
    def __init__(self,
                 task: str,
                 transforms: Optional[List[Transform]] = None,
                 data_folder: str = DATA_FOLDER,
                 test: bool = False) -> None:

        self.transforms = transforms
        arc = ARCDataset(tasks=[task], data_folder=data_folder)
        x_train, y_train, x_test, y_test = arc[task]
        if(test):
            self.x, self.y = x_test, y_test
        else:
            self.x, self.y = x_train, y_train

    @property
    def index(self) -> Sequence:
        """Every task can be transformed 'n' times.
        Index is cumulative sum of these values over episodes."""
        if(self.transforms is None):
            return np.arange(1, len(self)+1)

        if(getattr(self, '_index', None) is None):
            self._create_index()

        return self._index

    def _create_index(self) -> None:
        index = []
        for episode_id in range(len(self.x)):
            per_task_cnt = 0
            for tr in self.transforms:
                data = (self.x[episode_id], self.y[episode_id])
                per_task_cnt += tr.count(data)
            index.append(per_task_cnt)
        self._index = np.cumsum(index)

    def __repr__(self) -> str:
        n = len(self.x)
        n_transformed = len(self)
        return f'length transformed = {n_transformed}, length original = {n}'

    def __len__(self) -> int:
        if self.transforms:
            return self.index[-1]
        else:
            return len(self.x)

    def __getitem__(self, id: int) -> T:
        """Find task and make tranformations.

        Parameters
        ----------

        id : int
            Name or id of task.

        Returns
        -------

        T: Tuple[List[np.ndarray], List[np.ndarray]]
        """
        no_transform = self.transforms is None
        if(no_transform):
            return [self.x[id]], [self.y[id]]
        else:
            assert id < len(self)
            # find original task
            idx = int(np.searchsorted(self.index, id, 'right'))
            original_data: T = ([self.x[idx]], [self.y[idx]])

            # find id for transform
            start_idx = self.index[idx]
            tr_id = id - start_idx

            # make transforms
            data = original_data
            for t in self.transforms:
                c = t.count(data)
                q, r = divmod(tr_id, c)
                data = t.transform_data(r, data)
                tr_id = q
            return data


def plot_array(ax: Axes, data: np.ndarray, title: str) -> None:
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F011BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    ax.imshow(data, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x-0.5 for x in range(1+len(data))])
    ax.set_xticks([x-0.5 for x in range(1+len(data[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)


def plot_task(task: T, predictions: Optional[List[np.ndarray]] = None) -> None:
    x_train, y_train, x_test, y_test = task
    num_train = len(x_train)
    _, axs = plt.subplots(2, num_train, figsize=(3*num_train, 3*2))
    for i in range(num_train):
        plot_array(axs[0, i], x_train[i], 'train-input')
        plot_array(axs[1, i], y_train[i], 'train-output')
    plt.tight_layout()
    plt.show()

    num_test = len(x_test)
    _, axs = plt.subplots(2, num_test, figsize=(3*num_test, 3*2),
                          squeeze=False)

    for i in range(num_test):
        plot_array(axs[0, i], x_test[i], 'test-input')
        plot_array(axs[1, i], y_test[i], 'test-output')
    plt.tight_layout()
    plt.show()

    if(predictions):
        num_preds = len(predictions)
        _, axs = plt.subplots(2, num_preds, figsize=(3*num_preds, 3*2),
                              squeeze=False)

        for i in range(num_preds):
            plot_array(axs[0, i], x_test[i], 'test-input')
            plot_array(axs[1, i], predictions[i], 'test-prediction')
            plt.tight_layout()
            plt.show()


def plot_unfolded_task(task: T) -> None:
    x, y = task
    _, axs = plt.subplots(2, 1, figsize=(3*1, 3*2))
    plot_array(axs[0], x[0], 'input')
    plot_array(axs[1], y[0], 'output')
    plt.tight_layout()
    plt.show()
