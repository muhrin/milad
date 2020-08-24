# -*- coding: utf-8 -*-
"""Module that is concerned with the calculation of moment invariants"""
import ast
import pathlib
from typing import Sequence, Union, List

import numpy

from . import moments

__all__ = 'MomentInvariant', 'read_invariants', 'RES_DIR'

# The resources directory
RES_DIR = pathlib.Path(__file__).parent / 'res'
GEOMETRIC_INVARIANTS = RES_DIR / 'rot3dinvs8mat.txt'
COMPLEX_INVARIANTS = RES_DIR / 'cmfs7indep_0.txt'

import operator
from functools import reduce  # Required in Python 3


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class MomentInvariant:
    """Class for calculating a moment invariant based"""
    def __init__(self, weight):
        self._weight = weight
        self._terms = []
        self._max_order = 0

        self._farray = None
        self._indarray = None
        self._norm_power = None

    @property
    def weight(self):
        return self._weight

    @property
    def terms(self):
        return self._terms

    @property
    def max_order(self) -> int:
        return self._max_order

    def insert(self, prefactor, indices: Sequence):
        """
        :param prefactor: the prefactor for this term in the invariant
        :param indices: the indices of the moments involved in this invariant
        """
        if not all(len(entry) == 3 for entry in indices):
            raise ValueError(
                'There have to be three indices per entry, got: {}'.format(
                    indices))
        self._terms.append((prefactor, tuple(indices)))
        self._max_order = max(self._max_order, numpy.max(indices))

    def build(self):
        factors, arr = zip(*self._terms)
        # self._farray = numpy.array([term[0] for term in self._terms])
        # self._indarray = numpy.array([term[1] for term in self._terms])
        self._farray = numpy.asarray(factors)
        self._indarray = numpy.asarray(arr)
        term = self._terms[0][1]
        self._norm_power = numpy.sum(term) / 3. + len(term)

    def apply(self, raw_moments: numpy.ndarray, normalise=True) -> float:
        """Compute this invariant from the given moments optionally normalising"""

        if isinstance(raw_moments, numpy.ndarray):
            # This performs the above
            indices = self._indarray
            total = numpy.dot(
                self._farray,
                numpy.product(raw_moments[indices[:, :, 0], indices[:, :, 1],
                                          indices[:, :, 2]],
                              axis=1))
        else:
            # This is slower version of above but compatible with moments that aren't numpy arrays
            total = 0.
            for factor, indices in self._terms:
                product = 1.0
                for index in indices:
                    product *= raw_moments.moment(*index)
                total += factor * product

        if normalise:
            return total / raw_moments[0, 0, 0]**self._norm_power

        return total


def apply_invariants(invariants: List[MomentInvariant],
                     moms: numpy.array,
                     normalise=False) -> numpy.array:
    """Calculate the moment invariants for a given set of moments

    :param invariants: a list of invariants to calculate
    :param moms: the moments to use
    :param normalise: if True fill normalise the moments using the 0th moment
    """
    result = numpy.empty(len(invariants))
    for idx, invariant in enumerate(invariants):
        result[idx] = invariant.apply(moms, normalise=normalise)
    return result


def read_invariants(filename: str = GEOMETRIC_INVARIANTS,
                    read_max: int = None) -> List[MomentInvariant]:
    """Read the Flusser, Suk and Zitová invariants.

    :param filename: the filename to read from, default to geometric moments invariants
    :param read_max: the maximum number of invariants to read
    """
    invariants = []
    with open(filename, 'r') as file:

        for line in file:
            line = line.rstrip()
            if line:
                # New entry
                header = [int(number) for number in line.split(' ')]
                num_terms = header[2]
                invariant = MomentInvariant(num_terms)

                # Now read the actual terms
                line = file.readline().rstrip()
                while line:
                    # Use literal_eval to get complex number as well
                    terms = tuple(map(ast.literal_eval, line.split(' ')))
                    prefactor = terms[0]

                    indices = []
                    for idx in range(num_terms):
                        indices.append(
                            tuple(terms[idx * 3 + 1:(idx + 1) * 3 + 1]))
                    invariant.insert(prefactor, indices)

                    line = file.readline().rstrip()

                invariant.build()
                invariants.append(invariant)
                if len(invariants) == read_max:
                    break

    return invariants


def calc_moment_invariants(invariants: Sequence[MomentInvariant],
                           positions: numpy.array,
                           sigma: Union[float, numpy.array] = 0.4,
                           masses: Union[float, numpy.array] = 1.,
                           normalise=False) -> Sequence[float]:
    """Calculate the moment invariants for a set of Gaussians at the given positions."""
    max_order = 0

    # Calculate the maximum order invariant we'll need
    for inv in invariants:
        max_order = max(max_order, inv.max_order)

    raw_moments = moments.calc_raw_moments3d(max_order, positions, sigma,
                                             masses)
    return tuple(
        invariant.apply(raw_moments, normalise) for invariant in invariants)
