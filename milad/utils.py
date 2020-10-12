# -*- coding: utf-8 -*-
from typing import Iterator

import numpy as np

__all__ = 'generate_all_pair_distances'


def calculate_all_pair_distances(vectors, sort_result=True):
    """Calculate all pair distances between the given vectors"""
    num = len(vectors)
    lengths = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            dr = vectors[i] - vectors[j]
            lengths.append(np.linalg.norm(dr))

    if sort_result:
        lengths.sort()
    return lengths


def even(val: int) -> bool:
    """Test if an integer is event.  Returns True if so."""
    return (val % 2) == 0


def from_to(*args) -> Iterator[int]:
    """Like range() but inclusive of supper bound and automatically does iteration of ranges with a
    negative step e.g. 0, -4 will a range containing 0, -1, -2, -3, -4"""
    if len(args) not in (1, 2):
        raise ValueError('Takes one or two args, got: {}'.format(args))

    if len(args) == 1:
        start = 0
        stop = args[0]
    else:
        start = args[0]
        stop = args[1]

    ubound = stop + 1 if stop > 0 else stop - 1
    step = 1 if stop > 0 else -1
    return range(start, ubound, step)


class CoefficientCapture:

    class Capture:

        def __init__(self, mtx: np.array, idx):
            self._mtx = mtx
            self._idx = idx

        def __mul__(self, other):
            print('Coeff: {}, idx: {}'.format(other, self._idx))
            self._mtx[self._idx] = other
            return self

        def __rmul__(self, other):
            return self.__mul__(other)

        def __iadd__(self, other):
            self._mtx[self._idx] += other

        def __riadd__(self, other):
            return self.__iadd__(other)

    def __init__(self, shape: tuple):
        self._mtx = np.zeros(shape)

    def __getitem__(self, item):
        print('Capturing {}'.format(item))
        return self.Capture(self._mtx, item)

    @property
    def mtx(self) -> np.array:
        return self._mtx


def outer_product(*array) -> np.array:
    if not array:
        raise ValueError('No arrays supplied')

    product = array[0]
    for entry in array[1:]:
        product = np.tensordot(product, entry, axes=0)

    return product
