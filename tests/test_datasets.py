from itertools import permutations

import numpy as np

from datasets.arc import ARCDataset
from datasets.transforms import ColorPermutation

# ARC dataset tasks number
N = 800


def sample_check(arc_dataset: ARCDataset, id: int):
    sample = arc_dataset[id]
    x_train, y_train, x_test, y_test = sample
    assert type(x_train) is list
    assert type(x_train[0]) is np.ndarray
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def same_shape_check(sample1, sample2):
    x_train1, y_train1, x_test1, y_test1 = sample1
    x_train2, y_train2, x_test2, y_test2 = sample2

    zips = [zip(x_train1, x_train2), zip(y_train1, y_train2), zip(x_test1, x_test2), zip(y_test1, y_test2)]

    for z in zips:
        for ep in z:
            assert ep[0].shape == ep[1].shape


def same_sample_check(sample1, sample2):
    x_train1, y_train1, x_test1, y_test1 = sample1
    x_train2, y_train2, x_test2, y_test2 = sample2

    zips = [zip(x_train1, x_train2), zip(y_train1, y_train2), zip(x_test1, x_test2), zip(y_test1, y_test2)]

    for z in zips:
        for ep in z:
            assert np.allclose(ep[0], ep[1])


def test_arc_no_transforms():
    ds = ARCDataset(transforms=None)
    assert len(ds.tasks) == N
    assert len(ds) == N
    assert ds.transforms is None

    sample_check(ds, N-1)

    half = N//2
    half_tasks = ds.tasks[:half]
    ds_half = ARCDataset(transforms=None, tasks=half_tasks)
    assert len(ds_half.tasks) == half
    assert len(ds_half) == half
    sample_check(ds_half, half-1)
    

def test_transformations():
    cp = ColorPermutation(max_colors=10)
    n = 10
    r = 5
    i = 100
    expected = list(permutations(range(n),r))[i]
    testee = cp._get_permutations(n, r, i)
    assert expected == testee


def test_arc_transformations_cp():
    cp_10 = ColorPermutation(max_colors=10)
    ds_10 = ARCDataset(transforms=cp_10)
    assert len(ds_10.index) == len(ds_10.tasks)
    assert len(ds_10) == ds_10.index[-1]
    assert len(ds_10.original_dataset) == N
    sample_check(ds_10, N)

    cp_11 = ColorPermutation(max_colors=11)
    ds_11 = ARCDataset(transforms=cp_11)
    assert (len(ds_10.tasks) == N) and (len(ds_11.tasks) == N)
    assert (len(ds_10) > N) and (len(ds_11) > N)
    assert len(ds_11) > len(ds_10)

    #limit check
    cp_10_limit_1000 = ColorPermutation(max_colors=10, limit=1000)
    ds_10_limit_1000 = ARCDataset(transforms=cp_10_limit_1000)
    assert len(ds_10_limit_1000) < len(ds_10)
    sample_check(ds_10_limit_1000, N)

    cp_10_limit_1 = ColorPermutation(max_colors=10, limit=1)
    ds_10_limit_1 = ARCDataset(transforms=cp_10_limit_1)
    assert len(ds_10_limit_1) == N
    sample_check(ds_10_limit_1, N-1)


def test_index():
    cp = ColorPermutation(10, 1000)
    ds = ARCDataset(transforms=cp)

    # make shift
    sub = np.insert(ds.index, 0, 0)[:-1]
    per_task_cnt = ds.index - sub

    # check limit per task
    assert max(per_task_cnt) <= 1000

    for task_idx in range(N):
        transformed_idx = ds.index[task_idx]

        # check last trasform for task
        same_shape_check(ds[transformed_idx-1], ds.original_dataset[task_idx])

        # check first transform for task
        n_transforms = per_task_cnt[task_idx]
        same_shape_check(ds[transformed_idx-n_transforms], ds.original_dataset[task_idx])

