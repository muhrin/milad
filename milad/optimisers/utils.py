# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np


class Data:
    """Data used during an optimisation"""

    def __init__(self, use_jacobian, complex_input: bool, verbose=False, builder=None, callback: Callable = None):
        self.use_jacobian = use_jacobian
        self.complex_input = complex_input
        self.verbose = verbose
        self.builder = builder
        self.callback = callback
        self.iter = 0


def join_state(state: np.ndarray) -> np.ndarray:
    half_size = int(len(state) / 2)
    out_state = state[:half_size] + 1j * state[half_size:]
    return out_state


def split_state(state: np.ndarray) -> np.ndarray:
    """Split is a state vector that is (potentially) complex.  If it is the vector is double in rows to be
    (reals, imags), otherwise the vector is simply returned unaltered."""
    if np.iscomplexobj(state):
        if isinstance(state, np.ndarray):
            return np.concatenate((state.real, state.imag))
        return np.array((state.real, state.imag))

    return state


def split_jacobian(jacobian: np.ndarray, complex_inputs: bool, complex_outputs: bool) -> np.ndarray:
    """Assuming that the function is analytic split the Jacobian into parts corresponding to a combination of
    complex inputs/outputs.  The layout of the return Jacobian is:

    | d Re(f)   d Re(f) |
    | -------   ------- |
    | d Re(x)   d Im(x) |
    |                   |
    | d Im(f)   d Im(f) |
    | -------   ------- |
    | d Re(x)   d Im(x) |

    If the function has complex inputs and real outputs then only the first row is return.
    If the function has real inputs and complex outputs then only the first column is return.
    Otherwise the full Jacobian is returned.

    See the Cauchy-Riemann equations for more details:
    https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations
    """
    if not complex_inputs and not complex_outputs:
        return jacobian.real

    orig_size = jacobian.shape
    if len(orig_size) == 1:
        orig_size = (orig_size[0], 1)
    new_size = list(orig_size)
    if complex_outputs:
        new_size[0] *= 2
    if complex_inputs:
        new_size[1] *= 2

    jac = np.empty(new_size)
    # Input: Real, Output: Real
    jac[:orig_size[0], :orig_size[1]] = jacobian.real
    if complex_outputs:
        # Input: Real, Output: Imag
        jac[orig_size[0]:, :orig_size[1]] = jacobian.imag
    if complex_inputs:
        complex_part = jacobian * 1j
        # Input: Imag, Output: Real
        jac[:orig_size[0], orig_size[1]:] = complex_part.real
        if complex_outputs:
            # Input: Imag, Output: Imag
            jac[orig_size[0]:, orig_size[1]:] = complex_part.imag

    return jac
