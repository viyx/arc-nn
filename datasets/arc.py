import os
import glob
import json
import logging
from typing import Tuple, List, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes
from torch.utils.data import Dataset

from .transforms import Transform


logger = logging.getLogger('matplotlib')
logger.setLevel(logging.ERROR)

DATA_FOLDER = 'datasets/data'
ARC_TASK_FORMAT = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]


class ARCDataset(Dataset):
    """Base class with access to raw data. It also can make and store transofmations.
    
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

        if(transforms):
            self.original_dataset = ARCDataset(tasks=tasks,
                                               data_folder=data_folder)
        else:
            self.original_dataset = self
        self.transforms = transforms
        self._read_tasks(data_folder, tasks)

    def _read_tasks(self, folder: str, tasks: Optional[List[str]]) -> None:
            if(tasks is None):
                train_tasks = glob.glob(folder + '/training/*.json')
                eval_tasks = glob.glob(folder + '/evaluation/*.json')
                self.tasks = train_tasks + eval_tasks
            else:
                assert len(tasks) >= 0, 'Specify at least one task.'
                self.tasks = tasks

    @property
    def index(self) -> Optional[np.ndarray]:
        """Task can be transformed 'n' times.
        Index is cumulative sum of those values over tasks."""
        if(self.transforms is None):
            return None
        if(getattr(self, '_index', None) is None):
            self._create_index()
        return self._index

    def _create_index(self) -> None:
        index = []
        for task in self.tasks:
            per_task_cnt = 0
            for tr in self.transforms:
                per_task_cnt += tr.count(self.original_dataset[task])
            index.append(per_task_cnt)
        self._index = np.cumsum(index)

    def __repr__(self) -> str:
        n_train = len([t for t in self.tasks if 'train' in t])
        n_eval = len([t for t in self.tasks if 'evaluation' in t])
        n_test = len([t for t in self.tasks if 'test' in t])
        n_trasformed = len(self)

        return ('ARC folder distribution: all = {}, train = {}, eval = {}, test = {}. Transformed : {}'
                .format(len(self.tasks), n_train, n_eval, n_test, n_trasformed))

    def __len__(self) -> int:
        if self.transforms:
            return self.index[-1]
        else:
            return len(self.tasks)

    def __getitem__(self, id: Union[int, str]) -> ARC_TASK_FORMAT:
        """Find task and make tranformations.
        
        Parameters
        ----------

        id : {str, int}
            Name or id of task.

        Returns
        -------

        T: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]]]
        Provides `x_train, y_train, x_test, y_test` np.ndarrays in one tuple.
        `y_test` can be None in kaggle submission regime.
        """
        if(not self.transforms):
            if(type(id) is int):
                filename = self.tasks[id]
            else: 
                filename = id
            with open(filename) as f:
                sample = json.load(f)
                train_x = list(
                    map(lambda d: np.array(d['input']),  sample['train']))
                train_y = list(
                    map(lambda d: np.array(d['output']), sample['train']))
                test_x = list(
                    map(lambda d: np.array(d['input']),  sample['test']))
                if('output' in sample['test'][-1]):
                    test_y = list(map(lambda d: np.array(d['output']), sample['test']))
                else: test_y = None
                return train_x, train_y, test_x, test_y
        else:
            if(id is str):
                assert False, "Can't use taskname with tranforms."
            assert id < len(self)

            # find original task
            original_task_idx = int(np.searchsorted(self.index, id, 'right'))
            original_data = self.original_dataset[original_task_idx]
            # find id for transform
            start_idx = self.index[original_task_idx]
            tr_id = id - start_idx
            # make transforms
            data = original_data
            for t in self.transforms:
                c = t.count(data)
                q, r = divmod(tr_id, c)
                data = t.transform_data(q, data)
                tr_id = r
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

    x: 
    """
    def __init__(self,
        task: str,
        transforms: Optional[List[Transform]] = None,
        data_folder: str = DATA_FOLDER,
        test: bool = False) -> None:

        # if(transforms):
        #     self.original_dataset =\
        #         UnfoldARCDataset(task=task,
        #                          data_folder=data_folder,
        #                          transforms=None,
        #                          test=test)
        # else:
        #     self.original_dataset = self
        self.transforms = transforms
        arc = ARCDataset(tasks=[task], data_folder=data_folder)
        x_train, y_train, x_test, y_test = arc[task]
        if(test):
            self.x ,self.y = x_test, y_test
        else:
            self.x ,self.y = x_train, y_train

    @property
    def index(self) -> Optional[np.ndarray]:
        """Every task can be transformed 'n' times.
        Index is cumulative sum of these values over tasks."""
        if(self.transforms is None):
            return None
        
        if(getattr(self, '_index', None) is None):
            self._create_index()
        
        return self._index

    def _create_index(self) -> None:
        index = []
        for episode_id in range(len(self.x)):
            per_task_cnt = 0
            for tr in self.transforms:
                per_task_cnt += tr.count([self.x[episode_id], self.y[episode_id]])
            index.append(per_task_cnt)
        self._index = np.cumsum(index)

    def __repr__(self) -> str:
        n = len(self.original_dataset)
        n_transformed = len(self)
        return f'length transformed = {n_transformed}, length original = {n}'

    def __len__(self) -> int:
        if self.transforms:
            return self.index[-1]
        else:
            return len(self.x)

    def __getitem__(self, id: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Find task and make tranformations.
        
        Parameters
        ----------

        id : int
            Name or id of task.

        Returns
        -------

        T: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]
        Provides `x_train, y_train, x_test, y_test` np.ndarrays in one tuple.
        """
        no_transform = self.transforms is None
        if(no_transform):
            return [self.x[id]], [self.y[id]]
        else:
            assert id < len(self)
            # find original task
            idx = int(np.searchsorted(self.index, id, 'right'))
            original_data = [self.x[idx]], [self.y[idx]]

            # find id for transform
            start_idx = self.index[idx]
            tr_id = id - start_idx

            # make transforms
            data = original_data
            for t in self.transforms:
                c = t.count(data)
                q, r = divmod(tr_id, c)
                data = t.transform_data(q, data)
                tr_id = r
            return data


def plot_array(ax: Axes, data: np.ndarray, train_or_test: str, input_or_output: str) -> None:
    cmap = colors.ListedColormap(
        ['#037777777777', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F011BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=-1, vmax=9)
    
    ax.imshow(data, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=-1.5)    
    ax.set_yticks([x-1.5 for x in range(1+len(data))])
    ax.set_xticks([x-1.5 for x in range(1+len(data[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' ' + input_or_output)


def plot_task(task: ARC_TASK_FORMAT, predictions: Optional[List[np.ndarray]] = None) -> None:
    x_train, y_train, x_test, y_test = task
    num_train = len(x_train)
    _, axs = plt.subplots(1, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_array(axs[-1,i], x_train[i],'train','input')
        plot_array(axs[0,i], y_train[i],'train','output')
    plt.tight_layout()
    plt.show()
        
    num_test = len(x_test)
    _, axs = plt.subplots(1, num_test, figsize=(3*num_test,3*2), squeeze=False)
    
    for i in range(num_test):
        plot_array(axs[-1,i], x_test[i],'test','input')
        plot_array(axs[0,i], y_test[i],'test','output')  
    plt.tight_layout()
    plt.show()

    if(predictions):
        num_preds = len(predictions)
        _, axs = plt.subplots(1, num_preds, figsize=(3*num_preds,3*2), squeeze=False)
        
        for i in range(num_preds):
            plot_array(axs[-1,i], x_test[i],'test','input')
            plot_array(axs[0,i], predictions[i],'test','prediction')  
            plt.tight_layout()
            plt.show()