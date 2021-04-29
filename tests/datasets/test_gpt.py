import numpy as np
from hydra.experimental import initialize, compose
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from datasets.gpt import OneTaskOneDataset, OneTaskCollator, POSITION_PAD_TOKEN
from datasets.transforms import ColorPermutation
from datasets.arc import ARCDataset, UnfoldARCDataset


TASKS = ARCDataset().tasks
PAD_TOKEN = 13


def test_onetaskoneds():
    transform = [ColorPermutation(10, limit=1000)]
    for i in range(len(TASKS)):
        uarc = UnfoldARCDataset(task=TASKS[i],
                                transforms=transform)
        arc = ARCDataset(tasks=TASKS, transforms=transform)

        # test split_train_val=True
        ds = OneTaskOneDataset(transforms=[transform, None, None],
                            same_train_val=True,
                            task=TASKS[i],
                            split='(0.9, 0.1)')
        train, test, val = ds.train_test_val
        sum1 = len(train) + len(val)
        assert sum1 == len(uarc)

        _, _, x_test, _ = arc.original_dataset[i]
        assert len(test) == len(x_test)
        
        assert all([len(d[0]) == 2 for d in (train, test, val)])

        # test split_train_val=False
        transform2 = [[ColorPermutation(11)],
                    [ColorPermutation(10, limit=2)],
                    [ColorPermutation(11, limit=100)]]
        ds = OneTaskOneDataset(transforms=transform2,
                            same_train_val=False,
                            task=TASKS[i],
                            split='(0.9, 0.1)')
        train, test, val = ds.train_test_val
        assert len(train) > len(val)
        assert len(val) > len(test)


def test_one_task_collator() -> None:
    transform = ([ColorPermutation(10, limit=1000)],
                 None,
                 [ColorPermutation(10, limit=100)])
                 
    ds = OneTaskOneDataset(transforms=transform,
                           same_train_val=False,
                           task=TASKS[0],
                           split='(0.9, 0.1)')
    
    for _ds in ds.train_test_val:
        for add_pos_tokens in [True, False]:
            collator1 = OneTaskCollator(
            pad_token=PAD_TOKEN,
            end_line_token=12,
            add_pos_tokens=add_pos_tokens
            )
            dl = DataLoader(_ds, collate_fn=collator1, batch_size=32, drop_last=False)
            for x, y in dl:
                assert sum((x[:,-1])  == PAD_TOKEN) == 0
                assert sum((y[:,0])  == PAD_TOKEN) == 0

                if(add_pos_tokens):
                    size = x.size()
                    assert size[1] % 2 == 0
                    _x = x.view(size[0], 2, size[1] // 2)
                    tokens = _x[:, 0, :]
                    positions = _x[:, 1, :]
                    m1 = tokens == PAD_TOKEN
                    m2 = positions == POSITION_PAD_TOKEN
                    assert np.allclose(m1, m2)
