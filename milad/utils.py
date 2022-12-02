# -*- coding: utf-8 -*-
from typing import Iterator

import numpy as np

__all__ = ("calculate_all_pair_distances",)


def calculate_all_pair_distances(vectors, sort_result=True):
    """Calculate all pair distances between the given vectors"""
    num = len(vectors)
    lengths = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            dr = vectors[i] - vectors[j]  # pylint: disable=invalid-name
            lengths.append(np.linalg.norm(dr))

    if sort_result:
        lengths.sort()
    return lengths


def even(val: int) -> bool:
    """Test if an integer is event.  Returns True if so."""
    return (val % 2) == 0


def odd(val: int) -> bool:
    """Test if an integer is odd.  Returns True if so."""
    return not even(val)


def inclusive(*args) -> Iterator[int]:
    """Like range() but inclusive of upper bound and automatically does iteration of ranges with a
    negative step e.g. 0, -4 will produce a range containing 0, -1, -2, -3, -4"""
    if len(args) not in (1, 2, 3):
        raise ValueError("Takes one or two args, got: {}".format(args))

    if len(args) == 3:
        # Assume form is start, stop, step
        start, stop, step = args
    else:
        if len(args) == 1:
            start = 0
            stop = args[0]
        else:
            start = args[0]
            stop = args[1]

        step = 1 if start <= stop else -1

    sign = 1 if step > 0 else -1
    idx = start
    while sign * (stop - idx) >= 0:
        yield idx
        idx += step


def outer_product(*array) -> np.array:
    if not array:
        raise ValueError("No arrays supplied")

    product = array[0]
    for entry in array[1:]:
        product = np.tensordot(product, entry, axes=0)

    return product
