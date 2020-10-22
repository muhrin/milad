# -*- coding: utf-8 -*-
"""Module for calculating Zernike moments"""

import logging
import itertools
import numbers
import functools
from typing import Union, List, Tuple, Dict, Mapping, Any, Iterator

import numpy
import numpy as np
import scipy
import scipy.special
from sympy.core import symbol
import sympy

from . import base_moments
from . import functions
from . import geometric
from .utils import even, inclusive

_LOGGER = logging.getLogger(__name__)

# pylint: disable=invalid-name

__all__ = ('ZernikeMoments', 'from_deltas', 'from_gaussians')


def from_deltas(
    max_order: int, positions: numpy.array, weights: Union[numbers.Number, numpy.array] = 1.
) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a collection of delta functions with optional weights

    :param max_order: the order of Zernike moments to calculate to
    :param positions: the Cartesian positions of the delta functions
    :param weights: the weights of the delta functions, can be a scalar or numpy.array of the same
        length as positions
    """
    _domain_check(positions)
    geom_moments = geometric.from_deltas(16, positions, weights)
    return from_geometric_moments(max_order, geom_moments)


def from_gaussians(
    max_order: int,
    positions: numpy.ndarray,
    sigmas: Union[numbers.Number, numpy.array] = 0.1,
    weights: Union[numbers.Number, numpy.array] = 0.1
) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a collection of Gaussian functions with the given sigmas
    and weights

    :param max_order: the order of Zernike moments to calculate to
    :param positions: the Cartesian positions of the delta functions
    :param sigmas: the sigmas of the Gaussians, can be a scalar or numpy.array of the same length
        as positions
    :param weights: the weights of the delta functions, can be a scalar or numpy.array of the same
        length as positions
    """
    _domain_check(positions)
    geom_moments = geometric.from_gaussians(16, positions, sigmas, weights)
    return from_geometric_moments(max_order, geom_moments)


def from_geometric_moments(order: int, geom_moments: numpy.array) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a set of geometric moments

    :param order: the order of Zernike moments to calculate to
    :param geom_moments: the geometric moments
    """
    # Let's calculate all the moments (for positive m)
    omega = numpy.empty((order + 1, order + 1, order + 1), dtype=complex)

    for n, l, m in iter_indices(order):
        omega[n, l, m] = omega_nl_m(n, l, m, geom_moments)

    return ZernikeMoments(order, omega)


class ZernikeMoments(base_moments.Moments):
    """A container class for calculated Zernike moments"""

    @classmethod
    def linear_index(cls, index: base_moments.Index) -> int:
        """Given a triple of Zernike moment indices this will return the corresponding linearly ordered index"""
        return linear_index(index)

    @staticmethod
    def triple_index(linear_index: int, redundant=False) -> Tuple:
        """Get the triple index from the given lexicographic index"""
        for entry, _ in zip(iter_indices(redundant=redundant), range(linear_index + 1)):
            pass
        return entry

    @classmethod
    def from_vector(cls, n_max: int, vec: numpy.array) -> 'ZernikeMoments':
        moms = ZernikeMoments(n_max)

        for (n, l, m), value in zip(iter_indices(), vec):
            moms[n, l, m] = value

        return moms

    def __init__(self, n_max: int, omega: numpy.array = None):
        """Construct a Zernike moments object

        :param omega: an optional moments matrix to initialise the class with
        """
        self._max_n = n_max

        if omega is not None:
            self._moments = omega
        else:
            self._moments = numpy.empty((n_max + 1, n_max + 1, n_max + 1), dtype=complex)

            # Fill with a number I will recognise if it's still left there
            # (which it shouldn't be for valid indexes)
            self._moments.fill(float('nan'))

    @property
    def dtype(self):
        return complex

    @property
    def real(self):
        moms = ZernikeMoments(self._max_n)
        moms._moments = self._moments.real
        return moms

    @property
    def imag(self):
        moms = ZernikeMoments(self._max_n)
        moms._moments = self._moments.imag
        return moms

    @property
    def vector(self):
        """Return this set of moments as a vector"""
        return numpy.array([self[indices] for indices in self.iter_indices(redundant=True)])

    def __getitem__(self, item) -> complex:
        return self.moment(*item)

    def __setitem__(self, key, value: complex):
        if isinstance(key, int):
            # Assume the caller is treating us like a vector
            n, l, m = self.triple_index(key)
        else:
            n, l, m = key

        if m < 0:
            m = -m
            value = (-1)**m * value.conjugate()
        self._moments[n, l, m] = value

    def iter_indices(self, redundant=False):
        yield from iter_indices(max_order=self._max_n, redundant=redundant)

    def to_matrix(self) -> np.array:
        raise AttributeError('Zernike moments cannot be converted to a matrix as the orders use negative indexing')

    @staticmethod
    def num_moments(max_order: int, redundant=False) -> int:
        """Get the total number of Zernike moments up to the maximum order"""
        return sum(1 for _ in iter_indices(max_order, redundant=redundant))

    def moment(self, n: int, l: int, m: int) -> complex:
        """Get the n, l, m^th moment"""
        if m < 0:
            return (-1)**(-m) * self.moment(n, l, -m).conjugate()

        assert_valid(n, l, m)
        return self._moments[n, l, m]

        # m_prime = abs(m)
        # omega = self._moments[n, l, m_prime]
        # if m < 0:
        #     omega = (-1) ** m_prime * omega.conjugate()
        # return omega

    def value_at(self, x: numpy.array, order: int = None) -> float:
        """Reconstruct the value at x from the moments

        :param x: the point to get the value at
        :param order: the maximum order to go up to (defaults to the order of these moments)
        """
        order = order or self._max_n

        if len(x.shape) == 2:
            moms = [geometric.from_deltas(self._max_n, pt).to_matrix() for pt in x]
            query = numpy.empty(list(moms[0].shape) + [len(moms)])
            for idx, entry in enumerate(moms):
                query[:, :, :, idx] = entry
        else:
            query = geometric.from_deltas(self._max_n, [x])

        value = 0.0

        for n, l, m in iter_indices(max_order=order):
            omega = self.moment(n, l, m)
            z = sum_chi_nlm(n, l, m, query)
            value += omega * z

            if m != 0:
                # Now do the symmetric -m part
                z_conj = (-1)**m * z.conjugate()
                value += z_conj * self.moment(n, l, -m)

        return value.real


class ZernikeMomentCalculator(functions.Function):
    output_type = ZernikeMoments
    supports_jacobian = True
    dtype = complex

    def __init__(self, max_order: int):
        super().__init__()
        self._max_order = max_order
        self._geometric_moments_calculator = geometric.GeometricMomentsCalculator(max_order)
        self._jacobian = None

    def empty_output(self, in_state: functions.State) -> ZernikeMoments:
        return ZernikeMoments(self._max_order)

    def output_length(self, in_state: functions.State) -> int:
        return ZernikeMoments.num_moments(self._max_order, redundant=True)

    def evaluate(self,
                 state: functions.State,
                 get_jacobian=False) -> Union[ZernikeMoments, Tuple[ZernikeMoments, np.ndarray]]:
        moments = ZernikeMoments(self._max_order)
        geom_moments = self._geometric_moments_calculator(state, get_jacobian)
        geom_jac = None

        if get_jacobian:
            geom_moments, geom_jac = geom_moments

        for n, l, m in iter_indices(self._max_order):
            moments[n, l, m] = omega_nl_m(n, l, m, geom_moments.moments)

        if get_jacobian:
            jac = np.matmul(self._get_jacobian(), geom_jac)
            return moments, jac

        return moments

    def _get_jacobian(self):
        if self._jacobian is None:
            # Calculate the first time
            self._jacobian = get_jacobian_wrt_geom_moments(self._max_order)

        return self._jacobian


def get_jacobian_wrt_geom_moments(max_order: int):
    """Get the Jacobian of the Zernike moments wrt Geometric moments as input"""

    # Calculate
    class DerivativeTracker:

        def __init__(self, max_ord):
            self._max_order = max_ord
            self._num_coeffs = geometric.GeometricMoments.num_moments(max_ord)

        def __getitem__(self, item: base_moments.Index):
            if not isinstance(item, tuple):
                raise TypeError(item)

            vector = np.zeros(self._num_coeffs)
            try:
                vector[geometric.linear_index(self._max_order, item)] = 1.
            except IndexError:
                raise IndexError(f'index {item} out of bounds for geometric moments of order {self._max_order}')
            return vector

    O = max_order
    tracker = DerivativeTracker(O)
    num_moments = ZernikeMoments.num_moments(O, redundant=True)  # Number of zernike moments
    num_geometric_moments = geometric.GeometricMoments.num_moments(O)
    jacobian = np.zeros((num_moments, num_geometric_moments), dtype=complex)

    for idx, (n, l, m) in enumerate(iter_indices(O, redundant=True)):
        if m < 0:
            # First calculate the derivatives for all the positive m indices
            continue

        vec = omega_nl_m(n, l, m, tracker)
        jacobian[idx, :] = vec

        minus_m = (-1)**m * vec.conjugate()
        minus_m_idx = linear_index((n, l, -m))
        jacobian[minus_m_idx, :] = minus_m

    return jacobian


class SymbolicInvariants:

    def __init__(self, n_max):
        self._n_max = n_max
        self._c = symbol.symbols(f'c:{ZernikeMoments.num_moments(n_max)}')

    def symbols(self):
        return self._c

    def __getitem__(self, item):
        target_n, target_l, target_m = item

        count = 0
        for n in inclusive(self._n_max):
            for l in inclusive(n):
                if not even(n - l):
                    continue
                for m in inclusive(l):
                    if n == target_n and l == target_l and m == abs(target_m):
                        value = self._c[count]
                        if target_m < 0:
                            value = (-1)**m * value.conjugate()
                        return value
                    count += 1

        raise RuntimeError("Shouldn't get here")

    def moment(self, n: int, l: int, m: int):
        return self.__getitem__((n, l, m))


@functools.lru_cache(maxsize=None)
def factorial(n: int) -> int:
    return scipy.special.factorial(n, exact=True)


@functools.lru_cache(maxsize=None)
def binomial_coeff(n: int, k: int) -> int:
    return scipy.special.comb(n, k, exact=True)


@functools.lru_cache(maxsize=None)
def c_l_m(l: int, m: int) -> float:
    """Calculate the normalisation factor"""
    if m < 0:
        # make use of symmetry c_l^m == c_l^-m
        return c_l_m(l, -m)

    return ((2 * l + 1) * factorial(l + m) * factorial(l - m))**0.5 / factorial(l)


@functools.lru_cache(maxsize=None)
def q_kl_nu(k: int, l: int, nu: int) -> float:
    """Calculate the Zernike geometric moment conversion factors"""
    return (-1) ** k / 2 ** (2 * k) * \
           ((2 * l + 4 * k + 3) / 3) ** 0.5 * \
           binomial_coeff(2 * k, k) * \
           (-1) ** nu * \
           (binomial_coeff(k, nu) * binomial_coeff(2 * (k + l + nu) + 1, 2 * k)) / \
           binomial_coeff(k + l + nu, k)


def sum_chi_nlm(n: int, l: int, m: int, geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    """Calculate the Zernike geometric moment conversion factors"""
    return c_l_m(l, m) * 2**(-m) * sum1(n=n, l=l, m=m, geom_moments=geom_moments)


def sum1(n: int, l: int, m: int, geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    """Calculate the Zernike geometric moment conversion factors"""
    k = int((n - l) / 2)  # n - l is always even
    total = 0.
    for nu in inclusive(k):
        total += q_kl_nu(k, l, nu) * sum2(n=n, l=l, m=m, nu=nu, geom_moments=geom_moments)

    return total


def sum2(n: int, l: int, m: int, nu: int, geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for alpha in inclusive(nu):
        total += binomial_coeff(nu, alpha) * \
                 sum3(n=n, l=l, m=m, nu=nu, alpha=alpha, geom_moments=geom_moments)
    return total


def sum3(n: int, l: int, m: int, nu: int, alpha: int, geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for beta in inclusive(nu - alpha):
        total += binomial_coeff(nu - alpha, beta) * \
                 sum4(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, geom_moments=geom_moments)
    return total


def sum4(n: int, l: int, m: int, nu: int, alpha: int, beta: int,
         geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for u in inclusive(m):
        total += (-1) ** (m - u) * binomial_coeff(m, u) * 1j ** u * \
                 sum5(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u,
                      geom_moments=geom_moments)
    return total


def sum5(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int,
         geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for mu in inclusive(int((l - m) / 2.)):
        total += (-1) ** mu * 2 ** (-2 * mu) * binomial_coeff(l, mu) * \
                 binomial_coeff(l - mu, m + mu) * \
                 sum6(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u, mu=mu,
                      geom_moments=geom_moments)
    return total


def sum6(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int, mu: int,
         geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for v in inclusive(mu):
        r = 2 * (v + alpha) + u
        s = 2 * (mu - v + beta) + m - u
        t = 2 * (nu - alpha - beta - mu) + l - m

        if r + s + t > n:
            continue

        total += binomial_coeff(mu, v) * geom_moments[r, s, t]

    return total


def assert_valid(n: int, l: int, m: int):
    if not n >= 0:
        raise ValueError('n must be a positive integer')
    if not 0 <= l <= n:
        raise ValueError(f'l must be 0 <= l <= n, got n={n}, l={l}')
    if not -l <= m <= l:
        raise ValueError('m must be in the range -l <= m <= l')
    if not even(n - l):
        raise ValueError('n - l must be even')


def omega_nl_m(n: int, l: int, m: int, geom_moments: numpy.array) -> complex:
    """Given a set of geometric moments this function will compute the corresponding Zernike moments
    """
    assert_valid(n, l, m)
    if m < 0:
        # Symmetry relation
        return (-1)**(-m) * omega_nl_m(n, l, -m, geom_moments).conjugate()

    return 3. / (4. * np.pi) * sum_chi_nlm(n, l, m, geom_moments).conjugate()


def iter_indices(max_order: int = None, redundant=False) -> Iterator[Tuple]:
    """Iterate over Zernike function indices in lexicographic order.

    If redundant is True then all valid indices will be generated including those where there is a
    symmetry relation e.g.: o_22^-2 = (-1)^2 o22^2.

    :param max_order: order to provide indices up to, if None will generate all indices
    :param redundant: if True include redundant indices
    """
    upper = itertools.count() if max_order is None else inclusive(max_order)
    for n in upper:
        for l in inclusive(n):
            if not even(n - l):
                continue

            m_start = -l if redundant else 0
            for m in inclusive(m_start, l):
                yield n, l, m


@functools.lru_cache(maxsize=256)
def linear_index(index: base_moments.Index) -> int:
    """Given a triple of zernike function indices this will return a lexicographically ordered
    integer and a boolean indicating if (-1)**m conjugate(value) should be applied"""
    for linear, triple in enumerate(iter_indices(redundant=True)):
        if triple == index:
            return linear


def _domain_check(positions: numpy.array):
    """Check that a given set of positions are within the domain for which Zernike moments are
    defined i.e. that |r| <= 1."""
    for idx, pos in enumerate(positions):
        if numpy.dot(pos, pos) > 1.:
            _LOGGER.warning(
                'Delta function %i is outside of the domain of Zernike basis functions '
                'with are defined within the unit ball', idx
            )
