from itertools import permutations

from datasets.transforms import ColorPermutation
from datasets.arc import ARCDataset
from .utils import sample_check_arc, same_shape_check_arc, N


def test_color_permutation():
    # ColorPermutation
    cp = ColorPermutation(max_colors=10)
    n = 10
    r = 5
    i = 100
    expected = list(permutations(range(n), r))[i]
    testee = cp._get_permutations(n, r, i)
    assert expected == testee

    ds_10 = ARCDataset(transforms=[cp])
    assert ds_10.index is not None
    assert len(ds_10.index) == len(ds_10.tasks)
    assert len(ds_10) == ds_10.index[-1]
    assert len(ds_10.original_dataset) == N
    sample_check_arc(ds_10, N)
    ds = ARCDataset(transforms=None)
    same_shape_check_arc(ds[0], ds_10[0])

    cp_11 = ColorPermutation(max_colors=11)
    ds_11 = ARCDataset(transforms=[cp_11])
    assert (len(ds_10.tasks) == N) and (len(ds_11.tasks) == N)
    assert (len(ds_10) > N) and (len(ds_11) > N)
    assert len(ds_11) > len(ds_10)

    # limit check
    cp_10_limit_1000 = ColorPermutation(max_colors=10, limit=1000)
    ds_10_limit_1000 = ARCDataset(transforms=[cp_10_limit_1000])
    assert len(ds_10_limit_1000) < len(ds_10)
    sample_check_arc(ds_10_limit_1000, N)

    cp_10_limit_1 = ColorPermutation(max_colors=10, limit=1)
    ds_10_limit_1 = ARCDataset(transforms=[cp_10_limit_1])
    assert len(ds_10_limit_1) == N
    sample_check_arc(ds_10_limit_1, N-1)


# def test_unpacking():
    # up = UnPacking()
    # ds_up = ARCDataset(transforms=up)
    # ds = ARCDataset()

    # assert len(ds_up.index) == len(ds_up.tasks)
    # assert len(ds_up) == ds_up.index[-1]
    # assert len(ds_up.original_dataset) == N
    # assert len(ds_up) > len(ds_up.original_dataset)
    # x_train_up, y_train_up, *_ = ds_up[0]
    # x_train, y_train, *_ = ds[0]

    # assert len(x_train_up) == 1
    # assert len(y_train_up) == 1
    # assert x_train[0].shape == x_train_up[0].shape
    # assert y_train[0].shape == y_train_up[0].shape


# def test_all_transforms():
#     up = UnPacking()
#     cp = ColorPermutation(10)

#     ds_tr = ARCDataset(transforms=[up, cp])
#     ds = ARCDataset()

#     assert len(ds_tr) > len(ds)
