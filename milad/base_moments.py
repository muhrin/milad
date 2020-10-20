# -*- coding: utf-8 -*-
"""Abstract base classes and utilities for moments"""

import abc
from typing import Tuple

import numpy as np

from . import functions

Index = Tuple[int, int, int]  # A three dimensional moment index


class Moments(functions.State, metaclass=abc.ABCMeta):
    """A class representing three dimensional moments"""

    @property
    @abc.abstractmethod
    def dtype(self):
        """Get the number type of the moments (typically float or complex)"""

    @abc.abstractmethod
    def __getitem__(self, index: Index):
        """Get the moment of the given index"""

    def to_matrix(self) -> np.array:
        """Return the moments in a matrix.  Not all moment classes support this as they may have
        special indexing schemes in which case an AttributeError will be raised"""
        raise AttributeError('Does not support conversion to matrix')

    @abc.abstractmethod
    def moment(self, n: int, l: int, m: int):
        """Get the n, l, m^th moment"""

    @abc.abstractmethod
    def linear_index(self, index: Index) -> int:
        """Get a linearised index from the passed triple index"""

    @abc.abstractmethod
    def value_at(self, x: np.array, max_order: int = None):
        """Reconstruct the value at x from the moments

        :param x: the point to get the value at
        :param max_order: the maximum order to go up to (defaults to the maximum order of these
            moments)
        """
