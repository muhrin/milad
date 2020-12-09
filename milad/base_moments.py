# -*- coding: utf-8 -*-
"""Abstract base classes and utilities for moments"""

import abc
import collections
from typing import Tuple, List, Dict

import numpy as np

from . import functions

Index = Tuple[int, int, int]  # A three dimensional moment index


class Moments(functions.State, metaclass=abc.ABCMeta):
    """A class representing three dimensional moments"""

    # pylint: disable=invalid-name

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
