from itertools import permutations, product

from datasets.transforms import ColorPermutation
from datasets.arc import ARCDataset
from .utils import (sample_check_arc,
                    same_shape_check_arc,
                    same_figures_check,
                    same_sample_check,
                    N)


def test_color_permutation():
    nrange = range(2, 10)
    irange = [0,1]
    for (n, i) in product(nrange, irange):
        cp = ColorPermutation(max_colors=n)
        expected = list(permutations(range(n), n))[i]
        testee = cp._get_permutations(n, n, i)
        assert expected == testee

    cp = ColorPermutation(max_colors=10)
    ds_10 = ARCDataset(transforms=[cp])
    assert ds_10.index is not None
    assert len(ds_10.index) == len(ds_10.tasks)
    assert len(ds_10) == ds_10.index[-1]
    assert len(ds_10.original_dataset) == N
    sample_check_arc(ds_10, N)
    ds = ARCDataset(transforms=None)
    assert same_shape_check_arc(ds[0], ds_10[0])

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

    limit = 2
    cp_10_limit_2 = ColorPermutation(max_colors=10, limit=limit)
    ds_10_limit_2 = ARCDataset(transforms=[cp_10_limit_2])
    for i in range(N):
        orig = ds_10_limit_2.original_dataset[i]
        tr = ds_10_limit_2[i*limit+1] # second permutation
        assert same_figures_check(orig, tr)
        # assert not same_sample_check(orig, tr)