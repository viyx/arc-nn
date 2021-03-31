from torch.utils.data import Dataset
import numpy as np
import logging

from .abc import AbstractDataset
from .arc import ARCDataset
from .transforms import Transform

def flat_x(end_line_token, x):
    "Add column of `end_line` tokens and flat 2D array."
    assert x.ndim == 2
    x_pad = np.pad(x, ((0,0),(0,1)), 'constant', constant_values=end_line_token)
    x_ravel = x_pad.ravel()
    return x_ravel


def flat_xy(pair, promt_token, end_episode_token, end_line_token):
    "Flat x, y pairs and add `promt` and `end_episode` tokens"
    x, y = pair
    x = flat_x(x, end_episode_token)
    y = flat_x(y, end_episode_token)
    return np.concatenate([x, promt_token, y, end_episode_token], axis=None)
    

class GPTDataset(Dataset):
    """Flat 2D samples and add specials tokens in this way:
    
    flatten(x) + `promt` + flatten(y) + `end_episode`
    
    Here `flatten(arg)` is:
    flat 2D array and add `end_line` in the end of every line.
    """
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.n_colors = config.n_colors
        self.n_context = config.n_context
        self.padding = config.padding # expand x to n_context
        # max flatten y size with special tokens(30 end_lines,1 end_episode)
        self.target_length = config.target_length
        self.add_positions = config.add_positions

        
        # make special tokens 
        self.end_line_token =  config.end_line_token           # array of shape (10, 3) has 10 end_lines
        self.promt_token = config.promt_token                  # promt after every x
        self.end_episode_token = config.end_episode_token      # end of episode
        self.pad_token = config.pad_token                      # padding token (expand to n_context from the begining)

        # check similarity
        assert len(set([self.end_line_token, self.pad_token, self.promt_token, self.end_episode_token])) == 4
        
        # add 4 special token
        self.vocab_size = config.n_colors + 4
        
    def __len__(self):
        return len(self.dataset)
    

    # TODO not pass if no padding needed
    def pad(self, seq, to, direct, pad_token):
        "Pad sequence to left or right."
        x = np.array([pad_token] * to, dtype=np.long)
        if(direct == 'left'):
            x[-len(seq):] = seq
        if(direct == 'right'):
            x[:len(seq)] = seq
        return x

    def flat_all_sample(self, x, y, x_test, y_test):
        # flat train pairs
        tokens = [self.promt_token, self.end_episode_token, self.end_line_token]
        xy_train = list(map(lambda xy: flat_xy(xy, *tokens), zip(x, y)))
        xy_train = np.concatenate(xy_train, axis=None)
        
        #flat test pair
        
        # take only the first test episode (may be >1)
        # if we keep all test episodes batching would be harder to control
#         if(len(x_test) > 1):
        x_test = x_test[0]
        y_test = y_test[0]
            
        x_test = flat_x(x_test, self.end_line_token)
        y_test = flat_x(y_test, self.end_line_token)
        
        # just add end of episode
        y = np.concatenate([y_test, self.end_episode_token], axis=None)
        
        # pad y to max flattened 2D field
        if(len(y) < self.target_length and self.padding):
            y = self.pad(y, self.target_length, 'right', self.pad_token)
        
        # context: concat all
        # remove last token from y
        x = np.concatenate([xy_train, x_test, self.promt_token, y[:-1]], axis=None)
        
        # padding
        if(len(x) < self.n_context and self.padding): # expand sample to n_context
            x = self.pad(x, self.n_context, 'left', self.pad_token)
            
        # we dont make shift like x = data[:-1], y = data[1:] to decrease data flow
        # we will cut predictions to target_length in model in order to calculate criterion
        # len(x) = n_context, len(y) = target_length

        # tests
        # assert np.allclose(x[-self.target_length+1:], y[:-1])
        # assert (x[-self.target_length+1:] != self.pad_token).sum() > 2
        return x, y
    
    def __getitem__(self, id):
        "Get raw example, then flat it and add special symbols."
        x, y, x_test, y_test = self.dataset[id]
        

        # this ugly code adds position tokens
        # you can see result below
        # TODO move positinal tokens to ARCDataset
        if(self.add_positions):
            xy_train_pos = []
            xy_train_pos_ab = []
            for i in range(len(x)):
                x_ = ((x[i].shape[0])*(x[i].shape[1]+1))+1
                y_ = ((y[i].shape[0])*(y[i].shape[1]+1))+1
                xy_train_pos.extend(np.concatenate((np.arange(1,x_+1), np.arange(1,y_+1))))

                xy_train_pos_ab.extend([1]*x_)
                xy_train_pos_ab.extend([2]*y_)

            x_ = ((x_test[0].shape[0])*(x_test[0].shape[1]+1))+1
            y_ = ((y_test[0].shape[0])*(y_test[0].shape[1]+1))+1
            y_ = np.arange(1,y_+1)

            # pad y to max flattened 2D field
            if(len(y_) < self.target_length and self.padding):
                y_ = self.pad(y_, self.target_length, 'right', 0)
            xy_test_pos = np.concatenate((np.arange(1,x_+1), y_[:-1]))
            xy_test_pos_ab = []
            xy_test_pos_ab.extend([1]*x_)
            xy_test_pos_ab.extend([2]*(len(y_)-1))

            xy_train_pos.extend(xy_test_pos)

            # padding
            if(len(xy_train_pos) < self.n_context and self.padding):
                xy_train_pos = self.pad(xy_train_pos, self.n_context, 'left', 0)

            xy_train_pos_ab.extend(xy_test_pos_ab)
            # padding
            if(len(xy_train_pos_ab) < self.n_context and self.padding):
                xy_train_pos_ab = self.pad(xy_train_pos_ab, self.n_context, 'left', 0)


            # add positional tokens separately for each episode (multipling #tokens by 3)
            # example
            #                   |---------------------------x-----------------------|--------------------y----------------|
            # main tokens:      [<pad>, <pad>, <pad>, <pad>, <pad>, 5 ,7 ,6, <promt>, 8, 2, 3, <end_episode>, <pad>, <pad>]
            # pos tokens:       [0,     0,      0,      0,      0,  1 ,2 ,3,    4,    1, 2, 3,      4,           0,      0]
            # pos_ab tokens:    [0,     0,      0,      0,      0,  1 ,1 ,1,    1,    2, 2, 2,      2,           0,      0]
            
        x, y = self.flat_all_sample(x, y, x_test, y_test)
        if(self.add_positions):
            x = np.concatenate((x, xy_train_pos, xy_train_pos_ab), axis=None)

        return x, y


class MaxNDataset(AbstractDataset):
    "Fitler tasks by length of context."
    # TODO api for building datasets
    def __init__(
        self,
        config,
        transforms=[None, None, None],  # train, test, val transformations
        ):
        self.logger = logging.getLogger('MaxNDataset')
        self.maxn = config.maxn
        self.config = config
        self.target_length = config.target_length
        self.n_colors = config.n_colors
        self.padding = config.padding
        self.transforms = transforms
        # self.add_positions = config.add_positions
        super().__init__(config=config)

    def create_new_dataset(self):
        "Find tasks with length <= maxn. Create train, test, val datasets."

        # find tasks with length <= maxn
        ds = ARCDataset(self.config)
        gpt_ds = GPTDataset(ds, self.config)
        lxs = []

        for id in range(len(ds)):
            x_gpt, _ =  gpt_ds[id]
            lxs.append(len(x_gpt))
            
        lxs = pd.Series(lxs) # TODO replace pandas with np
        self.logger.info('Median length : {}'.format(lxs.median()))

        # multiply by 3 if GPTDataset adds position tokens
        # maxn = self.maxn if not self.add_positions else self.maxn * 3
        indices = lxs[lxs <= self.maxn].index.tolist()
        assert len(indices) > 0, f'No tasks with length {self.maxn}. Check --add_positions and --padding arguments.'
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
            arc = ARCDataset(tasks=tasks, transforms=transform)
            gpt = GPTDataset(arc, self.config)
            # with open(os.path.join(self.datadir, file), 'wb') as f:
                # pickle.dump(gpt, f)
            self.datasets.append(gpt)

        self.logger.info('Lengths after transforms. train : {}, test : {}, val : {}'.
            format(*map(len, self.datasets)))


# class OneTaskOneDataset(AbstractDataset):
#     def __init__(
#         self,
#         config,
#         transforms:list(Transform)=None
#         ):
#         self.logger = logging.getLogger('OneTaskOneDataset')
#         self.transforms = transforms
#         super().__init__(config=config)
    
#     class MiniDataset(Dataset):
#         def __init__(self, arc_dataset):
#             self.arc = arc_dataset

#         def __getitem__(self, idx):
            

#     def _create_new_dataset(self):
#         arc = ARCDataset(tasks=config.tasks, transforms=self.transforms)

