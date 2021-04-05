import numpy as np

from datasets.arc import ARCDataset, UnfoldARCDataset


# ARC dataset tasks number
N = 800


def sample_check_arc(ds: ARCDataset, id: int):
    sample = ds[id]
    x_train, y_train, x_test, y_test = sample
    assert type(x_train) is list
    assert type(x_train[0]) is np.ndarray
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def sample_check_onetask(ds: UnfoldARCDataset, id: int):
    sample = ds[id]
    x, y = sample
    assert type(x) is list
    assert type(x[0]) is np.ndarray
    assert len(x) == len(y)
    assert len(x) == 1


def same_shape_check_arc(sample1, sample2):
    x_train1, y_train1, x_test1, y_test1 = sample1
    x_train2, y_train2, x_test2, y_test2 = sample2

    zips = [zip(x_train1, x_train2), zip(y_train1, y_train2), zip(x_test1, x_test2), zip(y_test1, y_test2)]

    for z in zips:
        for ep in z:
            assert ep[0].shape == ep[1].shape


def same_shape_check_onetask(sample1, sample2):
    x1, y1 = sample1
    x2, y2 = sample2

    zips = [zip(x1, x2), zip(y1, y2)]

    for z in zips:
        for ep in z:
            assert ep[0].shape == ep[1].shape


# def same_sample_check(sample1, sample2):
#     x_train1, y_train1, x_test1, y_test1 = sample1
#     x_train2, y_train2, x_test2, y_test2 = sample2

#     zips = [zip(x_train1, x_train2), zip(y_train1, y_train2), zip(x_test1, x_test2), zip(y_test1, y_test2)]

#     for z in zips:
#         for ep in z:
#             assert np.allclose(ep[0], ep[1])