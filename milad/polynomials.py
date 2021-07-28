# -*- coding: utf-8 -*-
import abc
import collections
import numbers
from typing import Tuple, Dict, Set

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

    def __init__(
        self,
        degree: int,
        prefactors: np.ndarray = None,
        terms: np.ndarray = None,
        constant=0.,
        conjugate_values=False,
        simplify=True,
    ):
        """
        Create a new homogenous polynomial of a given degree

        :param degree: the degree of the polynomial
        :param prefactors: the array of prefactors
        :param terms: the array indices that make up this polynomial where the array is interpreted as
            [sum index, product index, value array index]
        :param constant: the additive constant
        """
        prefactors = np.array(prefactors)
        terms = np.array(terms)

        if terms.size == 0 and len(terms.shape) == 1:
            # Make terms the right shape if it is empty
            terms = np.empty((0, 0, 0))

        if len(terms.shape) != 3:
            raise ValueError(f'terms must be rank three array, got {len(terms.shape)}')
        if len(prefactors) != len(terms):
            raise ValueError(
                f'prefactors and terms must have same length, got prefactors={len(prefactors)}, terms={len(terms)}'
            )

        if simplify:
            prefactors, terms = self._simplify(prefactors, terms)

        self._degree = degree
        self._prefactors = prefactors
        self._terms = terms
        self._constant = constant
        self._conjugate = conjugate_values
        self._derivatives_cache = {}
        self._variables = None
        if self._terms.size > 0:
            self._evaluate_method = self._numpy_evaluate
        else:
            self._evaluate_method = lambda *args: self._constant

    def __mul__(self, value: numbers.Number):
        if isinstance(value, numbers.Number):
            if value == 1.:
                return self

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

        terms = [
            f"{prefactor} * x_{','.join(map(str, term))}" for prefactor, term in zip(self._prefactors, self._terms)
        ]
        if self._constant:
            terms.append(str(self._constant))

        return ' + '.join(terms)

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def prefactors(self) -> np.ndarray:
        return self._prefactors

    @property
    def terms(self) -> np.ndarray:
        return self._terms

    @property
    def constant(self):
        return self._constant

    @property
    def variables(self) -> Set[Tuple]:
        """Return a set of all the the indices used by these invariants"""
        if self._variables is None:
            indices = set()
            for product in self._terms:
                for idx in product:
                    indices.add(tuple(idx))
            self._variables = indices

        return self._variables

    def conjugate(self):
        """Return the complex conjugate of this polynomial"""
        return HomogenousPolynomial(
            self._degree,
            self._prefactors.conjugate() if self._prefactors is not None else None,
            self._terms if self._terms is not None else None,
            self._constant.conjugate(),
            conjugate_values=not self._conjugate
        )

    def evaluate(self, values: np.ndarray):
        if self._conjugate:
            values = values.conjugate()

        if isinstance(values, np.ndarray):
            return self._numpy_evaluate(values)

        return self._generic_evaluate(values)

    def _numpy_evaluate(self, values: np.ndarray):
        """Fast method to get the invariant from a numpy array"""
        total = self._constant  # type: float

        if self._terms.size > 0:
            if values.dtype == object or len(values.shape) > 3:
                total += numpy_evaluate(self.prefactors, self._terms, values)
            else:
                total += numba_evaluate(self.prefactors, self._terms, values)

        return total

    def _generic_evaluate(self, values):
        """Generic apply for values that support indexing.

        This is slower version of above but compatible with values that aren't numpy arrays"""
        total = self._constant
        for factor, indices in zip(self._prefactors, self._terms):
            product = 1
            for index in indices:
                product *= values[tuple(index)]
            total += factor * product
        return total

    __call__ = evaluate

    def get_partial_derivative(self, variable) -> 'HomogenousPolynomial':
        variable = tuple(variable)
        try:
            return self._derivatives_cache[variable]
        except KeyError:
            pass

        prefactors = []
        terms = []
        constant = 0

        for prefactor, product in zip(self._prefactors, self._terms):
            # Gather the terms in this product
            powers = collections.defaultdict(int)
            for indices in product:
                powers[tuple(indices)] += 1

            try:
                power = powers.pop(variable)
            except KeyError:
                # This term will not be in the derivative
                pass
            else:
                # Calculate the new prefactor
                new_prefactor = prefactor * power

                new_product = []
                # And add in the correct multiple of this variable
                if power != 1:
                    new_product.extend((variable,) * (power - 1))

                for var, power in powers.items():
                    # The other terms keep their exponent unchanged
                    new_product.extend((var,) * power)

                if power == 1 and len(product) == 1:
                    constant += new_prefactor
                else:
                    prefactors.append(new_prefactor)
                    terms.append(new_product)

        new_degree = self.degree - 1 if prefactors else 0
        deriv = HomogenousPolynomial(new_degree, prefactors, terms, constant, conjugate_values=self._conjugate)

        self._derivatives_cache[variable] = deriv
        return deriv

    def get_gradient(self) -> Dict[Tuple, 'HomogenousPolynomial']:
        return {variable: self.get_partial_derivative(variable) for variable in self.variables}

    @staticmethod
    def _simplify(prefactors: np.ndarray, terms: np.ndarray):
        new_prefactors = []
        new_terms = []
        for prefactor, product in zip(prefactors.tolist(), terms.tolist()):
            if prefactor == 0:
                continue

            found = False
            for i, new_term in enumerate(new_terms):
                if product == new_term:
                    new_prefactors[i] += prefactor
                    found = True
                    break

            if not found:
                new_prefactors.append(prefactor)
                new_terms.append(product)

        if new_prefactors:
            new_terms, new_prefactors = tuple(zip(*sorted(zip(new_terms, new_prefactors))))

        return np.array(new_prefactors), np.array(new_terms)


def numpy_evaluate(prefactors, indices: np.array, values: np.ndarray):
    """Fast method to get the polynomial from a numpy array"""
    return np.dot(prefactors, np.prod(values[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]], axis=1))


@numba.jit(parallel=False, nopython=True)
def numba_evaluate(prefactors, indices, moments):
    """Numba version to speed up calculation of larger polynomials"""
    total = 0.
    for idx in numba.prange(len(prefactors)):  # pylint: disable=not-an-iterable
        factor = prefactors[idx]
        entry = indices[idx]

        product = factor
        for index in entry:
            product *= moments[index[0], index[1], index[2]]

        total += product
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
            prefactors=np.array([1.]),
            terms=np.array([[list(indices)]]),
        )
