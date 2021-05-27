# -*- coding: utf-8 -*-
"""Module for things related to spherical harmonics"""
from typing import Union, Tuple, Optional

from . import utils

MaxOrRange = Union[int, Tuple[type(None), type(None)]]
Range = Tuple[int, int]

# pylint: disable=invalid-name

__all__ = 'IndexTraits', 'MaxOrRange', 'Range'


class IndexTraits:
    """Helper class to deal with various possible valid indices for spherical harmonics and a radial function"""

    def __init__(
        self, n_spec: MaxOrRange, l_spec: MaxOrRange = None, l_le_n: bool = False, n_minus_l_even: bool = False
    ):
        self._n_range = to_range(n_spec)
        self._l_range = to_range(l_spec or self._n_range[1])
        self._l_le_n = l_le_n
        self._n_minus_l_even = n_minus_l_even

    @property
    def n_range(self) -> Tuple[int, int]:
        return self._n_range

    @property
    def l_range(self) -> Tuple[int, int]:
        return self._l_range

    def iter_n(self, l: int, n_spec: MaxOrRange = None):
        """Iterate over all the valid n values for a given angular frequency, l"""
        n_min, n_max = make_range(n_spec, self._n_range)

        # If necessary, shift n_min up to the nearest higher or equal integer such that `even(n - l) is True`
        n_min = n_min + ((n_min - l) % 2) if self._n_minus_l_even else n_min
        yield from utils.inclusive(max(l, n_min) if self._l_le_n else 0, n_max, 2 if self._n_minus_l_even else 1)

    def iter_l(self, n: int, l_spec: MaxOrRange = None):
        """Iterate over all valid values of l for a given value of n"""
        l_min, l_max = make_range(l_spec, self._l_range)

        # If necessary, shift l_min up to the nearest higher or equal integer such that `even(n - l) is True`
        l_min = l_min + ((n - l_min) % 2) if self._n_minus_l_even else l_min
        yield from utils.inclusive(l_min, min(l_max, n) if self._l_le_n else l_max, 2 if self._n_minus_l_even else 1)

    def iter_nl(self, n_spec: MaxOrRange = None, l_spec: MaxOrRange = None):
        n_spec = make_range(n_spec, self._n_range)
        l_spec = make_range(l_spec, self._l_range)

        for n in utils.inclusive(*n_spec, 1):
            for l in self.iter_l(n, l_spec):
                yield n, l


def make_range(spec: Optional[MaxOrRange], default: Range) -> Range:
    if spec is None:
        return default

    val_range = list(to_range(spec))
    # Use our n_range as defaults if either min/max are empty
    val_range[0] = default[0] if val_range[0] is None else val_range[0]
    val_range[1] = default[1] if val_range[1] is None else val_range[1]

    return tuple(val_range)


def to_range(spec: MaxOrRange) -> Range:
    if isinstance(spec, tuple):
        return spec
    if isinstance(spec, int):
        return 0, spec

    raise ValueError(f"'{spec}' is not a valid value specification, takes integer (maximum) or 2-tuple (range)")
