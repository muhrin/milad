"""Module that is concerned with the calculation of moment invariants"""
import pathlib
from typing import Sequence

import numpy

from . import moments

# The resources directory
RES_DIR = pathlib.Path(__file__).parent / 'res'

__all__ = 'MomentInvariant', 'read_invariants'


class MomentInvariant:
    """Class for calculating a moment invariant based"""

    def __init__(self, weight):
        self._weight = weight
        self._terms = []

    @property
    def weight(self):
        return self._weight

    @property
    def terms(self):
        return self._terms

    def insert(self, prefactor, indices: Sequence):
        """
        :param prefactor: the prefactor for this term in the invariant
        :param indices: the indices of the moments involved in this invariant
        """
        self._terms.append((prefactor, tuple(indices)))

    def apply(self, raw_moments: numpy.array, normalise=True) -> float:
        """Compute this invariant from the given moments optioanlly normalising"""
        sum = 0.
        scale_factor = raw_moments[0, 0, 0] if normalise else 1.0

        total_factors = 0.
        for factor, indices in self._terms:
            total_factors += factor
            product = 1.0
            for index in indices:
                product *= raw_moments[index[0], index[1], index[2]] / scale_factor

            sum += factor * product

        return sum / total_factors


def read_invariants(filename: str = None, read_max: int = 100):
    """Read the Flusser, Suk and Zitová invariants.

    :param filename: the filename to read from, default to the rot3invs8mat file
    :param read_max: the maximum number of invariants to read
    """
    if filename is None:
        filename = RES_DIR / 'rot3dinvs8mat.txt'

    invariants = []
    with open(filename, 'r') as file:

        line = file.readline().rstrip()
        while line is not None:
            if line:
                # New entry
                header = [int(number) for number in line.split(" ")]
                num_terms = header[2]
                invariant = MomentInvariant(num_terms)

                # Now read the actual terms
                line = file.readline().rstrip()
                while line:
                    terms = [int(number) for number in line.split(" ")]
                    prefactor = terms[0]
                    indices = []
                    for idx in range(num_terms):
                        indices.append(numpy.array(terms[idx * 3 + 1: (idx + 1) * 3 + 1]))
                    invariant.insert(prefactor, indices)

                    line = file.readline().rstrip()

                invariants.append(invariant)
                if len(invariants) == read_max:
                    break

            line = file.readline().rstrip()

    return invariants


def calc_moment_invariants(
        invariants: Sequence[MomentInvariant],
        positions: numpy.array,
        sigma: float,
        max_order,
        scale=1.,
        normalise=True) -> Sequence[float]:
    """Calculate the moment invariants for a set of Gaussians at the given positions."""
    raw_moments = moments.calc_raw_moments3d(positions, sigma, max_order, scale, normalise)
    return tuple(invariant.apply(raw_moments) for invariant in invariants)
