# -*- coding: utf-8 -*-
"""Module for things related to spherical harmonics"""
import numbers
from typing import Union, Tuple, Optional, NamedTuple

import numpy as np
import numpy.ma

from . import utils

# pylint: disable=invalid-name


class InclusiveRange(NamedTuple):
    min: numbers.Number
    max: numbers.Number

    def __contains__(self, x: numbers.Number):
        return self.min <= x <= self.max

    def __len__(self):
        return (self.max - self.min) + 1


Range = InclusiveRange
RangeType = Union[Tuple[int, int], Range]
MaxOrRange = Union[int, Tuple[type(None), type(None)]]

__all__ = "IndexTraits", "MaxOrRange", "Range"


class IndexTraits:
    """Helper class to deal with various possible valid indices for spherical harmonics and a radial function"""

    def __init__(
        self,
        n_spec: MaxOrRange,
        l_spec: MaxOrRange = None,
        l_le_n: bool = False,
        n_minus_l_even: bool = False,
    ):
        self._n_range = to_range(n_spec)
        self._l_range = to_range(l_spec or self._n_range[1])
        self._l_le_n = l_le_n
        self._n_minus_l_even = n_minus_l_even

        if l_le_n and self._n_range[1] > self._n_range[1]:
            raise ValueError(
                f"Cannot have l_max > n_max ({self._n_range[1]} > {self._n_range[1]})"
            )

    @property
    def n(self) -> InclusiveRange:
        return self._n_range

    @property
    def N(self) -> int:
        """Return the total number of radial basis included"""
        return len(self._n_range)

    @property
    def l(self) -> InclusiveRange:  # noqa: E743
        return self._l_range

    @property
    def L(self) -> int:
        """Get the total number of angular frequencies included i.e. l = 0 -> L - 1"""
        return len(self._l_range)

    def iter_n(self, l: int, n_spec: MaxOrRange = None):
        """Iterate over all the valid n values for a given angular frequency, l"""
        n_min, n_max = make_range(n_spec, self._n_range)

        # If necessary, shift n_min up to the nearest higher or equal integer such that `even(n - l) is True`
        n_min = n_min + ((n_min - l) % 2) if self._n_minus_l_even else n_min
        yield from utils.inclusive(
            max(l, n_min) if self._l_le_n else 0,
            n_max,
            2 if self._n_minus_l_even else 1,
        )

    def iter_l(self, n: int, l_spec: MaxOrRange = None):
        """Iterate over all valid values of l for a given value of n"""
        l_min, l_max = make_range(l_spec, self._l_range)

        # If necessary, shift l_min up to the nearest higher or equal integer such that `even(n - l) is True`
        l_min = l_min + ((n - l_min) % 2) if self._n_minus_l_even else l_min
        yield from utils.inclusive(
            l_min,
            min(l_max, n) if self._l_le_n else l_max,
            2 if self._n_minus_l_even else 1,
        )

    @staticmethod
    def iter_m(l: int, m_spec: MaxOrRange = None):
        m_min, m_max = make_range(m_spec, (-l, l))
        yield from utils.inclusive(m_min, m_max)

    def iter_nl(self, n_spec: MaxOrRange = None, l_spec: MaxOrRange = None):
        n_spec = make_range(n_spec, self._n_range)
        l_spec = make_range(l_spec, self._l_range)

        for n in utils.inclusive(*n_spec, 1):
            for l in self.iter_l(n, l_spec):
                yield n, l

    def iter_nlm(
        self,
        n_spec: MaxOrRange = None,
        l_spec: MaxOrRange = None,
        m_spec: MaxOrRange = None,
    ):
        for n, l in self.iter_nl(n_spec, l_spec):
            for m in self.iter_m(l, m_spec):
                yield n, l, m

    def iter_lm(self, n: int, l_spec: MaxOrRange = None, m_spec: MaxOrRange = None):
        for l in self.iter_l(n, l_spec):
            for m in self.iter_m(l, m_spec):
                yield l, m

    __iter__ = iter_nlm


def make_range(spec: Optional[MaxOrRange], default: RangeType) -> Range:
    if spec is None:
        return default

    val_range = list(to_range(spec))
    # Use our n_range as defaults if either min/max are empty
    val_range[0] = default[0] if val_range[0] is None else val_range[0]
    val_range[1] = default[1] if val_range[1] is None else val_range[1]

    return Range(*val_range)


def to_range(spec: MaxOrRange) -> Range:
    if isinstance(spec, tuple):
        return Range(*spec)
    if isinstance(spec, (int, np.integer)):
        return Range(0, spec)

    raise ValueError(
        f"'{spec}' is not a valid value specification, takes integer (maximum) or 2-tuple (range)"
    )


def create_array(indices: IndexTraits, dtype=complex):
    """Create an empty array that is sized according to the passed indices"""
    array = np.empty((indices.N, indices.L, 2 * indices.L + 1), dtype=dtype)
    # Create the mask and unmask the valid elements
    mask = np.ones(array.shape, dtype=int)
    for idx in indices.iter_nlm():
        mask[idx] = 0
    return numpy.ma.array(array, mask=mask)
