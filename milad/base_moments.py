# -*- coding: utf-8 -*-
"""Abstract base classes and utilities for moments"""

import abc
import collections
from typing import Tuple, List, Dict, Iterator, Union

import numpy as np

from . import functions

Index = Tuple[int, int, int]  # A three dimensional moment index


class ReconstructionQuery:

    def __init__(self, max_order: int, points: np.ndarray):
        self._max_order = max_order
        self._points = points

    @property
    def max_order(self) -> int:
        return self._max_order

    @property
    def points(self):
        return self._points


class Moments(functions.State, metaclass=abc.ABCMeta):
    """A class representing three dimensional moments"""

    # pylint: disable=invalid-name

    @classmethod
    def from_indexed(cls, indexed, max_order: int, dtype=float) -> 'Moments':
        moments = cls(max_order, dtype=dtype)
        for idx in moments.iter_indices():
            moments[idx] = indexed.__getitem__(idx)
        return moments

    @property
    @abc.abstractmethod
    def dtype(self):
        """Get the number type of the moments (typically float or complex)"""

    @abc.abstractmethod
    def __getitem__(self, index: Index):
        """Get the moment of the given index"""

    @abc.abstractmethod
    def __setitem__(self, index: Union[int, Tuple], value):
        """Set the moment of the given index"""

    def to_matrix(self) -> np.array:  # pylint: disable=no-self-use
        """Return the moments in a matrix.  Not all moment classes support this as they may have
        special indexing schemes in which case an AttributeError will be raised"""
        raise AttributeError('Does not support conversion to matrix')

    @property
    @abc.abstractmethod
    def max_order(self):
        """Get the maximum order of the moments"""

    @abc.abstractmethod
    def moment(self, n: int, l: int, m: int):
        """Get the n, l, m^th moment"""

    @abc.abstractmethod
    def linear_index(self, index: Index) -> int:
        """Get a linearised index from the passed triple index"""

    @abc.abstractmethod
    def iter_indices(self) -> Iterator[Index]:
        """Iterate through the valid indices of these moments"""

    @abc.abstractmethod
    def value_at(self, x: np.array, max_order: int = None):
        """Reconstruct the value at x from the moments

        :param x: the point to get the value at
        :param max_order: the maximum order to go up to (defaults to the maximum order of these
            moments)
        """

    @abc.abstractmethod
    def get_mask(self, fill=None) -> 'Moments':
        """Get an empty set of moments all set to None that can be used as a mask to fix values e.g. for optimisation"""

    def grid_values(self, num_samples: int, zero_outside_domain=True):
        """Get a grid and corresponding values of the moments reconstructed at the gridpoints"""
        # Create a coordinate grid
        spacing = np.linspace(-1., 1., num_samples)
        grid = np.array(np.meshgrid(spacing, spacing, spacing))
        grid_points = grid.reshape(3, -1).T

        if zero_outside_domain:
            # Calculate the lengths squared and get the corresponding indexes
            length_sq = (grid_points**2).sum(axis=1)
            valid_idxs = np.argwhere(length_sq < 1)

            # Now calculate the grid values at those points, the rest are 0
            grid_vals = np.zeros(grid_points.shape[0])
            values = self.value_at(grid_points[valid_idxs][:, 0, :])
            np.put(grid_vals, valid_idxs, values, mode='raise')
        else:
            # Do all points, even those outside the domain
            grid_vals = self.value_at(grid_points)

        # Reshape into n * n * n array
        return grid, grid_vals.reshape((grid.shape[1:]))

    def reconstruct(self, query: ReconstructionQuery, order=None, zero_outside_domain=True):  # pylint: disable=unused-argument
        return self.value_at(query.points)

    @classmethod
    def get_grid(cls, num_samples: int, restrict_to_domain=True) -> np.ndarray:
        # Create a coordinate grid
        spacing = np.linspace(-1., 1., num_samples)
        grid = np.array(np.meshgrid(spacing, spacing, spacing))
        grid_points = grid.reshape(3, -1).T

        if restrict_to_domain:
            # Calculate the lengths squared and get the corresponding indexes
            length_sq = (grid_points**2).sum(axis=1)
            valid_idxs = np.argwhere(length_sq < 1)
            grid_points = grid_points[valid_idxs][:, 0, :]

        return grid_points

    @classmethod
    def create_reconstruction_query_from_grid(
        cls, order: int, num_sample: int, restrict_to_domain=True
    ) -> ReconstructionQuery:
        grid = cls.get_grid(num_sample, restrict_to_domain=restrict_to_domain)
        return cls.create_reconstruction_query(grid, order)

    @classmethod
    def create_reconstruction_query(cls, points: np.ndarray, order: int) -> ReconstructionQuery:
        return ReconstructionQuery(order, points)


ProductTerm = collections.namedtuple('ProductTerm', 'index terms')
Product = collections.namedtuple('Product', 'coeff terms')


class MomentsPolynomial:
    """Represents a polyonimal of moments.  The terms are products of moments with a prefactor"""
    __slots__ = ('_terms',)

    def __init__(self):
        self._terms: List[Product] = []

    def __str__(self):
        sum_parts = []
        for prefactor, product in self._terms:
            powers = self.collect_powers(product)
            product_parts = [str(prefactor)]
            product_parts.extend(
                'm{},{},{}^{}'.format(indices[0], indices[1], indices[2], power) for indices, power in powers.items()
            )

            string = ' '.join(product_parts)
            sum_parts.append(string)

        return ' + '.join(sum_parts)

    def append(self, term: Product):
        """Append a term to the polynomial"""
        self._terms.append(term)

    def evaluate(self, moments):
        """Evaluate the polynomial for a given set of moments"""
        total = 0.

        for term in self._terms:
            partial = term.coeff
            for product_term in term.terms:
                partial *= moments[product_term]
            total += partial

        return total

    @staticmethod
    def collect_powers(product: List[Tuple]) -> Dict[Tuple, int]:
        powers = collections.defaultdict(int)
        for indices in product:
            powers[indices] += 1
        return powers


class ProductBuilder:
    """Helper to build a product of moments that can form part of a moments polynomial"""

    def __init__(self, coeff):
        self._coeff = coeff
        self._terms = []

    def add(self, index):
        """Add a moment to the product using its index"""
        self._terms.append(index)

    def build(self) -> Product:
        """Build the product into a fixed tuple"""
        return Product(self._coeff, tuple(self._terms))
