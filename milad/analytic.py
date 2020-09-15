# -*- coding: utf-8 -*-
"""Module for analytic operations"""
import itertools
from typing import Tuple, Union

import sympy


def create_array(indexable, shape: Union[int, Tuple]) -> sympy.MutableDenseNDimArray:
    """Given an indexed object this will create an array of the specified shape containing the
    indexable object indexed at each position"""
    if isinstance(shape, int):
        shape = (shape,)

    components = sympy.MutableDenseNDimArray.zeros(*shape)

    ranges = [range(entry) for entry in shape]
    for indices in itertools.product(*ranges):
        components[indices] = indexable[indices]

    return components
