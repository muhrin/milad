# -*- coding: utf-8 -*-
import numpy as np

from milad import functions
from milad import base_moments

__all__ = ("Invariants",)


class Invariants(functions.Function):
    # pylint: disable=abstract-method
    input_type = base_moments.Moments
    output_type = np.ndarray
