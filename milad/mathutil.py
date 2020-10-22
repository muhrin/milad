# -*- coding: utf-8 -*-
import numpy as np


def to_real(vec: np.array) -> np.array:
    """Convert a given numpy vector containing complex numbers to one twice as long containing
    only real numbers where the first half contains the real and the second half the imaginary parts
    """
    view = vec.view('(2,)float')
    real = view.reshape(view.shape[0] * view.shape[1])
    return real


def to_complex(vec: np.array) -> np.array:
    """Given a vector of real numbers convert it to one half the size containing complex numbers
    where the first half of the original vector is treated as the real parts while the second half
    is used for the imaginary"""
    half_size = int(vec.size / 2)
    reshaped = vec.reshape((half_size, 2))
    view = reshaped.view(dtype=complex).reshape(half_size)

    return view


def even(val: int) -> bool:
    """Test if an integer is event.  Returns True if so."""
    return (val % 2) == 0


#
# class ComplexMatrix:
#     """A matrix that represents a complex number using real numbers.  It has the following form:
#
#     (a  b)
#     (-b a)
#
#     which represents z = a + ib.
#     """
#     __slots__ = ('_mtx',)
#
#     def __init__(self, mtx: np.ndarray):
#         self._mtx = mtx or np.empty((2, 2), dtype=float)
#
#     def __getitem__(self, item):
#
#
#     def __mul__(self, other) -> 'ComplexMatrix':
#         return ComplexMatrix(np.matmul(self._mtx, other._mtx))
#
#     def __add__(self, other) -> 'ComplexMatrix':
#         return ComplexMatrix(self._mtx + other._mtx)
#
#     def __sub__(self, other):
#         return ComplexMatrix(self._mtx - other._mtx)
