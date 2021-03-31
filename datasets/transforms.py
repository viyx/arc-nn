from math import factorial
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List

import numpy as np


arc_task_format = Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]]]


class Transform(ABC):
    "Abstract class which all transformation classes should derive."
    def __init__(self, limit: int = None):
        self.limit = limit

    @abstractmethod
    def count(self, data: arc_task_format) -> int:
        "Return number of all possible transformations for single task. Can't exceed the limit."
        raise NotImplementedError

    @abstractmethod
    def transform_data(self, idx: int, data: arc_task_format) -> arc_task_format:
        "Return transformation by idx from all possible ones."
        raise NotImplementedError
    

class ColorPermutation(Transform):
    "Replace colors in task."
    # TODO make shuffling for permutation when limit exists because
    # TODO now we take first #limit permutations, it shifts color distribution to first colors (0,1,2,3..)
    def __init__(self, max_colors: int, limit: Optional[int] = np.inf) -> None:
        self.max_colors = max_colors
        # if limit is None: limit = np.inf
        super().__init__(limit)

    @lru_cache()
    def P(self, n: int, r: int) -> int:
        "Return permutation counts. Order matters."
        return factorial(n) // factorial(n - r)

    def count(self, data: arc_task_format) -> int:
        """Number of permutations per one task.
        Find #unique colors in task and then count all permutations. P(n,r).
        """
        u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
        c = self.P(self.max_colors, len(u_colors))
        return min(self.limit, c)

    # may be removed when net is large and time to perform the operation ~ batch forwarding time
    # cache ~ 1GB per dataloader process
    # @lru_cache()
    def _get_permutations(self, n: int, r: int, idx: int) -> Tuple[int]:
        "Fast and low-memory analog of `list(permutations(range(n),r))[idx]`"
        
        result = set()
        all = set(range(n))

        for d in range(r):
            c = self.P(n-d,r-d)
            q1 = c//(n-d)
            q2, r2 = divmod(idx, q1)
            to_add = list(all-result)[q2]
            result |= {to_add}
            idx = r2
        result = tuple(result)

        # ground_truth = list(permutations(range(n),r))[idx]
        # assert result == ground_truth
        return result

    def transform_data(self, idx:int , data: arc_task_format) -> arc_task_format:
        """Get permutated data.
        Find unique colors in data, then estimate permutation for them
        and replace data with the permutation.
        """
        assert idx < self.limit, "Id should be less or equal then limit. Check caller function."
        u_colors = set.union(*[set(np.concatenate(i, axis=None)) for i in data])
        u_colors = np.array(list(u_colors))

        # p is tuple like (1,5,6)
        p = self._get_permutations(self.max_colors, len(u_colors), idx)

        # replace unique colors with permuted ones
        m = np.zeros(self.max_colors, dtype=np.long)
        m[u_colors] = p

        # replace color in all data
        p_data = tuple([[m[ep] for ep in d] for d in data])
        return p_data


