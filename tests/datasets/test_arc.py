import numpy as np
from typing import List

from datasets.arc import ARCDataset, UnfoldARCDataset
from datasets.transforms import ColorPermutation
from .utils import (sample_check_arc,
                    same_shape_check_arc,
                    same_shape_check_onetask,
                    sample_check_onetask,
                    N)


TASKNAME = 'datasets/data/evaluation/0a1d4ef5.json'


def test_arc():
    ds = ARCDataset(transforms=None)
    assert len(ds.tasks) == N
    assert len(ds) == N
    assert ds.transforms is None

    sample_check_arc(ds, N-1)

    half = N//2
    half_tasks = ds.tasks[:half]
    ds_half = ARCDataset(transforms=None, tasks=half_tasks)
    assert len(ds_half.tasks) == half
    assert len(ds_half) == half
    sample_check_arc(ds_half, half-1)


def check_unfoldarc(ds: UnfoldARCDataset, TASKNAME: str, test: bool) -> None:
    arc = ARCDataset(transforms=None)
    x_train, y_train, x_test, y_test = arc[TASKNAME]
    if(test):
        x_target = x_test
        y_target = y_test
    else:
        x_target = x_train
        y_target = y_train
    assert len(ds) == len(x_target)
    assert len(ds.x) == len(x_target)
    assert len(ds.x) == len(ds.y)
    assert len(ds[0]) == 2
    sample_check_onetask(ds, 0)


def test_unfoldarc():
    arc = ARCDataset(transforms=None)

    for test in [True, False]:
        ds = UnfoldARCDataset(task=TASKNAME,
                              transforms=None,
                              test=test)
        check_unfoldarc(ds, TASKNAME, test)


def test_arc_index():
    limit = 1000
    cp = ColorPermutation(10, limit=limit)
    ds = ARCDataset(transforms=[cp])

    # make shift
    sub = np.insert(ds.index, 0, 0)[:-1]
    per_task_cnt = ds.index - sub

    # check limit per task
    assert max(per_task_cnt) <= limit
    assert len(ds.index) == len(ds.original_dataset)

    for task_idx in range(N):
        transformed_idx = ds.index[task_idx]

        # check last trasform for task
        assert same_shape_check_arc(ds[transformed_idx-1], ds.original_dataset[task_idx])

        # check first transform for task
        n_transforms = per_task_cnt[task_idx]
        assert same_shape_check_arc(ds[transformed_idx-n_transforms], ds.original_dataset[task_idx])


def check_unfoldarc_index(ds: UnfoldARCDataset,
                          original_x: List[np.ndarray],
                          original_y: List[np.ndarray]) -> None:
    # make shift
    sub = np.insert(ds.index, 0, 0)[:-1]
    per_task_cnt = ds.index - sub
    # check limit per task
    limit = ds.transforms[0].limit
    assert max(per_task_cnt) <= limit
    assert len(ds.index) == len(original_x)

    for ep_idx in range(len(original_x)):
        original_task = [[original_x[ep_idx]], [original_y[ep_idx]]]
        transformed_idx = ds.index[ep_idx]

        # check last trasform for task
        same_shape_check_onetask(ds[transformed_idx-1], original_task)

        # check first transform for task
        n_transforms = per_task_cnt[ep_idx]
        same_shape_check_onetask(ds[transformed_idx-n_transforms], original_task)


def test_unfoldarc_index():
    limit = 1000
    cp = ColorPermutation(10, limit=limit)
    arc = ARCDataset(tasks=[TASKNAME])
    x_train, y_train, x_test, y_test = arc[TASKNAME]

    for test in [True, False]:
        ds = UnfoldARCDataset(task=TASKNAME, transforms=[cp], test=test)

        if(not test):
            original_x = x_train
            original_y = y_train
            check_unfoldarc_index(ds, original_x, original_y)
        else:
            original_x = x_test
            original_y = y_test
            check_unfoldarc_index(ds, original_x, original_y)
