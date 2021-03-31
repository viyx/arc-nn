import os
import glob
import json
from typing import Tuple, List, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.axes import Axes

from .transforms import Transform


arc_task_format = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]]]


class ARCDataset:
    """Base class with access to raw data. It also can make and store transofmations.
    
    Parameters
    ----------
    tasks : List[str], default=None
        List of tasks's file names.
        If None, they will be read from the `data` folder.

    transforms : List[Transform], default=None
        List of transformations for augmentation.

    data : str, default='data'
        Name of folder with ARC data.

    Attributes
    ----------

    original_dataset : ARCDataset
        Dataset without transformations.
    """
    def __init__(self,
        tasks: Optional[List[str]] = None,
        transforms: Optional[List[Transform]] = None,
        data_folder: str = 'datasets/data') -> None:

        if(transforms):
            self.original_dataset = ARCDataset(tasks=tasks, data_folder=data_folder, transforms=None)
            if(type(transforms) is not list): 
                transforms = [transforms]
        else:
            self.original_dataset = self
        self.transforms = transforms
        self.index = None
        
        # when start with `hydra` cwd is different
        # cwd = os.path.dirname(os.path.realpath(__file__))
        # data_folder = os.path.join(cwd, data_folder)
        if(tasks is None): # load all tasks
            train_tasks = glob.glob(data_folder + '/training/*.json')
            eval_tasks = glob.glob(data_folder + '/evaluation/*.json')
            self.tasks = train_tasks + eval_tasks
        else:
            assert len(tasks) >= 0, 'Specify at least one task.'
            self.tasks = tasks

        # create index
        if transforms:
            index = []
            for task in self.tasks:
                per_task_cnt = 0
                for tr in transforms:
                    per_task_cnt += tr.count(self.original_dataset[task])
                index.append(per_task_cnt)
            
            self.index = np.cumsum(index)

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

    def __getitem__(self, id: Union[int, str]) -> arc_task_format:
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
        no_transform = self.transforms is None

        # id is str, untransformed task
        if(no_transform):
            if(type(id) is int):
                #replace task id by its name
                id = self.tasks[id]
            with open(id) as f:
                sample = json.load(f)
            # map json to lists of 2D arrays
                train_x = list(
                    map(lambda d: np.array(d['input']),  sample['train']))
                train_y = list(
                    map(lambda d: np.array(d['output']), sample['train']))
                test_x = list(
                    map(lambda d: np.array(d['input']),  sample['test']))
                if('output' in sample['test'][-1]):
                    test_y = list(map(lambda d: np.array(d['output']), sample['test']))
                else: test_y = None

                # list(np.array(ndim=2)), list(np.array(ndim=2)) ...
                return train_x, train_y, test_x, test_y
        else:
            assert id < len(self)

            # find original task
            original_task_idx = int(np.searchsorted(self.index, id, 'right'))
            original_data = self.original_dataset[original_task_idx]

            # skip previous ids
            id -= self.index[original_task_idx]

            data = original_data
            # make transforms
            for t in self.transforms:
                c = t.count(data)
                q, r = divmod(id, c)
                data = t.transform_data(q, data)
                id = r

            return data

    def plot(self, id: int, predictions: Union[List[np.ndarray], None] = None) -> None:
        def plot_one(ax: Axes, data: np.ndarray, train_or_test: str, input_or_output: str) -> None:
            cmap = colors.ListedColormap(
                ['#037777777777', '#0074D9','#FF4136','#2ECC40','#FFDC00',
                '#AAAAAA', '#F011BE', '#FF851B', '#7FDBFF', '#870C25'])
            norm = colors.Normalize(vmin=-1, vmax=9)
            
            ax.imshow(data, cmap=cmap, norm=norm)
            ax.grid(True,which='both',color='lightgrey', linewidth=-1.5)    
            ax.set_yticks([x-1.5 for x in range(1+len(data))])
            ax.set_xticks([x-1.5 for x in range(1+len(data[0]))])     
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(train_or_test + ' ' + input_or_output)

        def plot_task(task: arc_task_format) -> None:
            """
            Plots the first train and test pairs of a specified task,
            using same color scheme as the ARC app
            """    
            x_train, y_train, x_test, y_test = task
            num_train = len(x_train)
            _, axs = plt.subplots(1, num_train, figsize=(3*num_train,3*2))
            for i in range(num_train):     
                plot_one(axs[-1,i], x_train[i],'train','input')
                plot_one(axs[0,i], y_train[i],'train','output')
            plt.tight_layout()
            plt.show()
                
            num_test = len(x_test)
            _, axs = plt.subplots(1, num_test, figsize=(3*num_test,3*2), squeeze=False)
            
            for i in range(num_test):
                plot_one(axs[-1,i], x_test[i],'test','input')
                plot_one(axs[0,i], y_test[i],'test','output')  
            plt.tight_layout()
            plt.show()

            if(predictions):
                num_preds = len(predictions)
                _, axs = plt.subplots(1, num_preds, figsize=(3*num_preds,3*2), squeeze=False)
                
                for i in range(num_preds):
                    plot_one(axs[-1,i], x_test[i],'test','input')
                    plot_one(axs[0,i], predictions[i],'test','prediction')  
                    plt.tight_layout()
                    plt.show()

        item = self[id]
        plot_task(item)
        
