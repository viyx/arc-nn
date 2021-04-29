from math import factorial
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


T = Tuple[List[np.ndarray], ...]


class Transform(ABC):
    "Abstract class which all transformation classes should derive."
    def __init__(self, limit: int):
        self.limit = limit

    @abstractmethod
    def count(self, data: T) -> int:
        """Return number of all possible transformations for single task.
        Can't exceed the limit."""
        raise NotImplementedError

    @abstractmethod
    def transform_data(self, idx: int, data: T) -> T:
        "Return transformation by idx from all possible ones."
        raise NotImplementedError


class ColorPermutation(Transform):
    "Replace colors in task."
    # TODO start permutations from original colors
    def __init__(self, max_colors: int, limit: int = np.inf) -> None:
        self.max_colors = max_colors
        super().__init__(limit)

    @lru_cache()
    def P(self, n: int, r: int) -> int:
        "Return permutation counts. Order matters."
        return factorial(n) // factorial(n - r)

    def count(self, taks_data: T) -> int:
        """Number of permutations per one task.
        Find #unique colors in task and then count all permutations. P(n,r).
        """
        u_colors = set.union(*[set(np.concatenate(i, axis=None))
                               for i in taks_data])
        c = self.P(self.max_colors, len(u_colors))
        return min(self.limit, c)

    # cache ~ 1GB per dataloader process
    # @lru_cache()
    def _get_permutations(self, n: int, r: int, idx: int) -> Tuple[int, ...]:
        "Fast and low-memory analog of `list(permutations(range(n),r))[idx]`"
        result: Tuple[int, ...] = tuple()
        all = set(range(n))

        for d in range(r):
            c = self.P(n-d, r-d)
            q1 = c//(n-d)
            q2, r2 = divmod(idx, q1)
            to_add = sorted(list(all-set(result)))[q2]
            result += (to_add,)
            idx = r2

        return result

    def transform_data(self, idx: int, taks_data: T) -> T:
        """Get permutated data.
        Find unique colors in data, then estimate permutation for them
        and replace data with the permutation.
        """
        assert idx < self.limit
        u_colors = set.union(*[set(np.concatenate(i, axis=None))
                               for i in taks_data])
        u_colors = np.array(list(u_colors))

        # p is tuple like (1,5,6)
        p = self._get_permutations(self.max_colors, len(u_colors), idx)

        # create mapping array
        m = np.zeros(self.max_colors, dtype=np.long)
        m[u_colors] = p

        # replace color in all taks_data
        p_data = tuple([[m[ep] for ep in d] for d in taks_data])
        return p_data
