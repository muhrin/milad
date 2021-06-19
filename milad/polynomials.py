# -*- coding: utf-8 -*-
import abc
import numbers

import numpy as np

import numba


class Polynomial(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def evaluate(self, values: np.ndarray):
        """Evaluate the polynomial with given set of values"""


class HomogenousPolynomial(Polynomial):
    """A homogenous polynomial.

    These are not strictly homogenous as they are allowed to have a numerical additive
    constant e.g. 5 x_0,0,0 + 2.  This makes them easier to work with and a better fit for our use case.
    """

    def __init__(self, degree: int, prefactors: np.ndarray = None, terms: np.ndarray = None, constant=0., real=True):
        """
        Create a new homogenous polynomial of a given degree

        :param degree: the degree of the polynomial
        :param prefactors: the set of prefactors
        :param terms: the set of array indices that make up this polynomial
        :param real: the variables (and therefore value) of this polynomial are real
        """
        self._degree = degree
        self._prefactors = prefactors
        self._terms = terms
        self._constant = constant
        self._real = real

    def __mul__(self, value: numbers.Number):
        if isinstance(value, numbers.Number):
            return HomogenousPolynomial(
                self._degree, self._prefactors * value if self._prefactors is not None else None,
                self._terms if self._terms is not None else None, self._constant * value
            )

        raise TypeError(f'Unexpected type {value.__class__.__name__}')

    __rmul__ = __mul__

    def __add__(self, other):
        if other == 0.:
            return self

        if not isinstance(other, HomogenousPolynomial):
            raise TypeError(f'Unexpected type {other.__class__.__name__}')

        if other._degree != self._degree:
            return ValueError(
                f'Cannot add two homogenous polynomials of different degree ({self.degree} vs {other.degree})'
            )

        return HomogenousPolynomial(
            self.degree,
            prefactors=np.concatenate((self._prefactors, other._prefactors)),
            terms=np.concatenate((self._terms, other._terms)),
            constant=self._constant + other.constant
        )

    __radd__ = __add__

    def __str__(self) -> str:
        if self._prefactors is None:
            return 'None'

        return ' + '.join([
            f"{prefactor} * x_{','.join(map(str, term))}" for prefactor, term in zip(self._prefactors, self._terms)
        ])

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def prefactors(self) -> np.ndarray:
        return self._prefactors

    @property
    def indices(self) -> np.ndarray:
        return self._terms

    @property
    def constant(self):
        return self._constant

    def conjugate(self):
        """Return the complex conjugate of this polynomial"""
        if self._real:
            return HomogenousPolynomial(
                self._degree,
                self._prefactors.conjugate() if self._prefactors is not None else None,
                self._terms if self._terms is not None else None, self._constant.conjugate()
            )

        raise RuntimeError("Haven't implemented conjugates of complex polynomials yet")

    def evaluate(self, values: np.ndarray):
        if isinstance(values, np.ndarray):
            return self._numpy_evaluate(values)

        return self._generic_evaluate(values)

    def _numpy_evaluate(self, values: np.ndarray):
        """Fast method to get the invariant from a numpy array"""
        total = self._constant  # type: float

        if self._terms is not None:
            if values.dtype == object:
                total += numpy_apply(self.prefactors, self._terms, values)
            else:
                total += numba_apply(self.prefactors, self._terms, values)

        return total

    def _generic_evaluate(self, values):
        """Generic apply for moments that support indexing.

        This is slower version of above but compatible with values that aren't numpy arrays"""
        total = self._constant
        for factor, indices in zip(self._prefactors, self._terms):
            product = 1
            for index in indices:
                product *= values[tuple(index)]
            total += factor * product
        return total

    __call__ = evaluate


def numpy_apply(prefactors, indices: np.array, values: np.ndarray):
    """Fast method to get the polynomial from a numpy array"""
    total = 0.
    total += np.dot(prefactors, np.prod(values[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]], axis=1))

    return total


@numba.jit(parallel=False, nopython=True)
def numba_apply(prefactors, indices, moments):
    """Numba version to speed up calculation of larger polynomials"""
    total = 0.
    for idx in numba.prange(len(prefactors)):  # pylint: disable=not-an-iterable
        factor = prefactors[idx]
        entry = indices[idx]

        product = 1.
        for index in entry:
            product *= moments[index[0], index[1], index[2]]

        total += factor * product
    return total


class PolyBuilder:
    """A convenience builder that will start a polynomial using __getitem__ so clients can just use:
        m = PolyBuilder()
        5 * m[3, 1, 3]
    to start a polynomial
    """

    def __getitem__(self, indices):
        return HomogenousPolynomial(
            1,
            prefactors=np.array([1]),
            terms=np.array([[list(indices)]]),
            real=True,
        )
