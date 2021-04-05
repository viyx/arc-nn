from hydra.experimental import initialize, compose
from omegaconf import DictConfig

from datasets.gpt import OneTaskOneDataset
from datasets.transforms import ColorPermutation
from datasets.arc import ARCDataset, UnfoldARCDataset


def test_onetaskonenn():
    with initialize(config_path="../../conf"):
        params = ['dataset=gpt/onetaskonenn',
                  "tasks=['datasets/data/training/0a938d79.json']"]
        cfg = compose(config_name="config", overrides=params)
        transform = [ColorPermutation(10, limit=1000)]
        uarc = UnfoldARCDataset(task=cfg.tasks[0],
                                transforms=transform)
        arc = ARCDataset(tasks=cfg.tasks,
                         transforms=transform)

        # test split_train_val=True
        ds = OneTaskOneDataset(config=cfg,
                               transforms=[transform, None, None],
                               same_train_val=True)
        train, test, val = ds.train_test_val
        sum1 = len(train) + len(val)
        assert sum1 == len(uarc)

        _, _, x_test, _ = arc[0]
        assert len(test) == len(x_test)
        
        assert all([len(d[0]) == 2 for d in (train, test, val)])

        # test split_train_val=False
        transform2 = [[ColorPermutation(10)],
                      [ColorPermutation(10, limit=2)],
                      [ColorPermutation(10, limit=100)]]
        ds = OneTaskOneDataset(cfg,
                               transforms=transform2,
                               same_train_val=False)
        train, test, val = ds.train_test_val
        assert len(train) > len(val)
        assert len(val) > len(test)