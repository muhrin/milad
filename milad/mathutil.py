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
