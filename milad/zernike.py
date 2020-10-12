# -*- coding: utf-8 -*-
"""Module for calculating Zernike moments"""

import logging
import itertools
import numbers
import operator
import functools
from typing import Union, List, Tuple, Dict, Mapping, Any, Iterator

import numpy
import numpy as np
import scipy
import scipy.special

from . import base_moments
from . import geometric
from . import mathutil
from . import invariants
from .utils import even, from_to

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

    @staticmethod
    def lexicographic_index(n: int, l: int, m: int) -> Tuple[int, bool]:
        """Given a triple of zernike function indices this will return a lexicographically ordered
        integer and a boolean indicating if (-1)**m conjugate(value) should be applied"""
        MAX_N = 100000
        idx = 0
        conjugate = m < 0
        m = abs(m)
        for n_ in from_to(MAX_N):
            for l_ in from_to(n_):
                if not even(n_ - l_):
                    continue

                for m_ in from_to(l_):
                    if n == n_ and l == l_ and m == m_:
                        return idx, conjugate

                    idx += 1

    @staticmethod
    def triple_index(lexicographic_index: int) -> Tuple:
        """Get the triple index from the given lexicographic index"""
        MAX_N = 100000
        counted = 0
        for n in from_to(MAX_N):
            for l in from_to(n):
                if not even(n - l):
                    continue
                for m in from_to(l):
                    if counted == lexicographic_index:
                        return n, l, m
                    counted += 1

        raise RuntimeError('Index too high: {}'.format(lexicographic_index))

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
        yield from iter_indices(order=self._max_n, redundant=redundant)

    def to_vector(self, redundant=False):
        """Return this set of moments as a vector"""
        return numpy.array([self[indices] for indices in self.iter_indices(redundant)])

    def to_matrix(self) -> np.array:
        raise AttributeError('Zernike moments cannot be converted to a matrix as the orders use negative indexing')

    @staticmethod
    def num_coeffs(order: int, redundant=False) -> int:
        """Get the total number of terms for Zernike coefficients up to the given order
        """
        return sum(1 for _ in iter_indices(order, redundant=redundant))

    def moment(self, n: int, l: int, m: int) -> complex:
        """Get the n, l, m^th moment"""
        assert_valid(n, l, m)
        m_prime = abs(m)
        omega = self._moments[n, l, m_prime]
        if m < 0:
            omega = (-1)**m_prime * omega.conjugate()
        return omega

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

        for n, l, m in iter_indices(order=order):
            omega = self.moment(n, l, m)
            z = sum_chi_nlm(n, l, m, query)
            value += omega * z

            if m != 0:
                # Now do the symmetric -m part
                z_conj = (-1)**m * z.conjugate()
                value += z_conj * self.moment(n, l, -m)

        return value.real


class SphericalInvariantsFunction:

    def __init__(self, invs: List[invariants.MomentInvariant]):
        self._invariants = invs
        self._max_n = max(inv.max_order for inv in invs)

    def __call__(self, arg: numpy.array) -> numpy.array:
        moms = ZernikeMoments(self._max_n)

        input_length = len(arg)
        for idx in range(input_length):
            moms[idx] = arg[idx]

        result = invariants.apply_invariants(self._invariants, moms)
        return result
        # return result[:input_length]

    def num_coeffs(self) -> int:
        return ZernikeMoments.num_coeffs(self._max_n)


from sympy.core import symbol


class SymbolicInvariants:

    def __init__(self, n_max):
        self._n_max = n_max
        self._c = symbol.symbols(f'c:{ZernikeMoments.num_coeffs(n_max)}')

    def symbols(self):
        return self._c

    def __getitem__(self, item):
        target_n, target_l, target_m = item

        count = 0
        for n in from_to(self._n_max):
            for l in from_to(n):
                if not even(n - l):
                    continue
                for m in from_to(l):
                    if n == target_n and l == target_l and m == abs(target_m):
                        value = self._c[count]
                        if target_m < 0:
                            value = (-1)**m * value.conjugate()
                        return value
                    count += 1

        raise RuntimeError("Shouldn't get here")

    def moment(self, n: int, l: int, m: int):
        return self.__getitem__((n, l, m))


class Residuals:

    def __init__(self, order: int, invs: invariants.MomentInvariants, fixed: Mapping[Tuple, Any] = None):
        self._order = order
        self._invariants = invs
        self._fixed_values = []

        if fixed:
            for (l, n, m), value in fixed.items():
                idx, conjugate = ZernikeMoments.lexicographic_index(l, n, m)
                if conjugate:
                    value = (-1)**(-m) * value.conjugate()
                self._fixed_values.append((idx, value))
                self._fixed_values.sort(key=operator.itemgetter(0))

        self._num_coeffs = ZernikeMoments.num_coeffs(self._order) - len(self._fixed_values)
        self._index_mapping = self._get_index_mapping()

        print('Looking for:')
        for indices in self._index_mapping.values():
            print('o{}'.format(indices))

    @property
    def vector_length(self) -> int:
        # Times by two because we treat real and imaginary parts separately
        return 2 * self._num_coeffs

    def to_moments(self, vec: np.array) -> ZernikeMoments:
        """Convert a prediction vector back to Zernike moments"""
        vec = mathutil.to_complex(vec)  # Convert the vector back to the original complex format

        if self._fixed_values:
            # Combine the fixed values and the current array
            combined = list(vec)
            for insert_at, value in self._fixed_values:
                combined.insert(insert_at, value)
            vec = np.array(combined)

        return ZernikeMoments.from_vector(self._order, vec)

    def residuals(self, predicted: np.array, data: np.array, epsilon: float = 1.):
        predicted_moms = self.to_moments(predicted)
        model = self._invariants.apply(predicted_moms, normalise=False)

        diff = (data - model)
        print(np.abs(diff.imag).max())

        diff = mathutil.to_real(diff)

        return diff / epsilon

    def jacobian(self, predicted: np.array):
        predicted_moms = self.to_moments(predicted)

        jacobian = np.zeros((len(self._invariants) * 2, len(predicted)))

        num_invs = len(self._invariants)

        for phi_idx, invariant in enumerate(self._invariants):
            derivatives = invariant.derivatives()
            for moment_index in range(self._num_coeffs):
                try:
                    indices = self._index_mapping[moment_index]
                    derivative = derivatives[indices]
                except KeyError:
                    pass
                else:
                    value = derivative.apply(predicted_moms)
                    wrt_real = 1 * value
                    wrt_imag = 1j * value

                    real_phi_real_c = wrt_real.real
                    real_phi_imag_c = wrt_imag.real

                    imag_phi_real_c = wrt_real.imag
                    imag_phi_imag_c = wrt_imag.imag

                    jacobian[phi_idx, moment_index] = real_phi_real_c
                    jacobian[phi_idx, self._num_coeffs + moment_index] = real_phi_imag_c

                    jacobian[phi_idx + num_invs, moment_index] = imag_phi_real_c
                    jacobian[phi_idx + num_invs, self._num_coeffs + moment_index] = imag_phi_imag_c

        return jacobian

    def _get_index_mapping(self) -> Dict[int, Tuple]:
        lexical_indices = list(range(self._num_coeffs))
        if self._fixed_values:
            # Combine the fixed values and the current array
            for insert_at, _value in self._fixed_values:
                lexical_indices.insert(insert_at, None)

        mapping = {}
        for idx, entry_idx in enumerate(lexical_indices):
            if entry_idx is not None:
                mapping[entry_idx] = ZernikeMoments.triple_index(idx)

        return mapping


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
    for nu in from_to(k):
        total += q_kl_nu(k, l, nu) * sum2(n=n, l=l, m=m, nu=nu, geom_moments=geom_moments)

    return total


def sum2(n: int, l: int, m: int, nu: int, geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for alpha in from_to(nu):
        total += binomial_coeff(nu, alpha) * \
                 sum3(n=n, l=l, m=m, nu=nu, alpha=alpha, geom_moments=geom_moments)
    return total


def sum3(n: int, l: int, m: int, nu: int, alpha: int, geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for beta in from_to(nu - alpha):
        total += binomial_coeff(nu - alpha, beta) * \
                 sum4(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, geom_moments=geom_moments)
    return total


def sum4(n: int, l: int, m: int, nu: int, alpha: int, beta: int,
         geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for u in from_to(m):
        total += (-1) ** (m - u) * binomial_coeff(m, u) * 1j ** u * \
                 sum5(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u,
                      geom_moments=geom_moments)
    return total


def sum5(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int,
         geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for mu in from_to(int((l - m) / 2.)):
        total += (-1) ** mu * 2 ** (-2 * mu) * binomial_coeff(l, mu) * \
                 binomial_coeff(l - mu, m + mu) * \
                 sum6(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u, mu=mu,
                      geom_moments=geom_moments)
    return total


def sum6(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int, mu: int,
         geom_moments: numpy.ndarray) -> Union[complex, numpy.array]:
    total = 0.
    for v in from_to(mu):
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
        raise ValueError('l must be 0 <= l <= n')
    if not -l <= m <= l:
        raise ValueError('m must be in the range -l <= m <= l')
    if not even(n - l):
        raise ValueError('n - l must be even')


def omega_nl_m(n: int, l: int, m: int, geom_moments: numpy.array) -> complex:
    """Given a set of geometric moments this function will compute the corresponding Zernike moments
    """
    assert_valid(n, l, m)
    return 3 / (4 * np.pi) * sum_chi_nlm(n, l, m, geom_moments).conjugate()


def iter_indices(order: int = None, redundant=False) -> Iterator[Tuple]:
    """Iterate over Zernike function indices in lexicographic order.

    If redundant is True then all valid indices will be generated including those where there is a
    symmetry relation e.g.: o_22^-2 = (-1)^2 o22^2.

    :param order: order to provide indices up to, if None will generate all indices
    :param redundant: if True include redundant indices
    """
    upper = itertools.count() if order is None else from_to(order)
    for n in upper:
        for l in from_to(n):
            if not even(n - l):
                continue

            m_start = -l if redundant else 0
            for m in from_to(m_start, l):
                yield n, l, m


def _domain_check(positions: numpy.array):
    """Check that a given set of positions are within the domain for which Zernike moments are
    defined i.e. that |r| <= 1."""
    for idx, pos in enumerate(positions):
        if numpy.dot(pos, pos) > 1.:
            _LOGGER.warning(
                'Delta function %i is outside of the domain of Zernike basis functions '
                'with are defined within the unit ball', idx
            )
