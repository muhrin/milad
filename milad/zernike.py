# -*- coding: utf-8 -*-
"""Module for calculating Zernike moments"""

import collections
import logging
import itertools
import numbers
import functools
from typing import Union, Tuple, Iterator, Dict

import numpy as np
import scipy
import scipy.special
from sympy.core import symbol

from . import base_moments
from . import functions
from . import geometric
from .utils import even, inclusive

_LOGGER = logging.getLogger(__name__)

# pylint: disable=invalid-name

__all__ = 'ZernikeMoments', 'from_deltas', 'from_gaussians'


def from_deltas(max_order: int, positions: np.array, weights: Union[numbers.Number, np.array] = 1.) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a collection of delta functions with optional weights

    :param max_order: the order of Zernike moments to calculate to
    :param positions: the Cartesian positions of the delta functions
    :param weights: the weights of the delta functions, can be a scalar or np.array of the same
        length as positions
    """
    _domain_check(positions)
    geom_moments = geometric.from_deltas(max_order, positions, weights)
    return from_geometric_moments(max_order, geom_moments)


def from_gaussians(
    max_order: int,
    positions: np.ndarray,
    sigmas: Union[numbers.Number, np.array] = 0.1,
    weights: Union[numbers.Number, np.array] = 0.1
) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a collection of Gaussian functions with the given sigmas
    and weights

    :param max_order: the order of Zernike moments to calculate to
    :param positions: the Cartesian positions of the delta functions
    :param sigmas: the sigmas of the Gaussians, can be a scalar or np.array of the same length
        as positions
    :param weights: the weights of the delta functions, can be a scalar or np.array of the same
        length as positions
    """
    _domain_check(positions)
    geom_moments = geometric.from_gaussians(max_order, positions, sigmas, weights)
    return from_geometric_moments(max_order, geom_moments)


def from_geometric_moments(max_order: int, geom_moments: np.array) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a set of geometric moments

    :param max_order: the order of Zernike moments to calculate to
    :param geom_moments: the geometric moments
    """
    return ZernikeMomentCalculator(max_order)(geom_moments)


class ZernikeReconstructionQuery(base_moments.ReconstructionQuery):

    def __init__(self, points: np.ndarray, moments: np.ndarray):
        super().__init__(points)
        self._moments = moments
        self._valid_idxs = None
        self._moments_in_domain = None

    @property
    def moments(self) -> np.ndarray:
        return self._moments

    @property
    def moments_in_domain(self) -> np.ndarray:
        if self._moments_in_domain is None:
            # Annoying indexing and unindexing, this returns a copy
            self._moments_in_domain = self._moments[:, :, :, self.valid_idxs][:, :, :, :, 0]
        return self._moments_in_domain

    @property
    def valid_idxs(self):
        """Get the indexes of points that are within the domain"""
        if self._valid_idxs is None:
            self._valid_idxs = ZernikeMoments._get_indices_in_domain(self.points)
        return self._valid_idxs


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
    def from_vector(cls, n_max: int, vec: np.array) -> 'ZernikeMoments':
        moms = ZernikeMoments(n_max)

        for (n, l, m), value in zip(iter_indices(), vec):
            moms[n, l, m] = value

        return moms

    def __init__(self, n_max: int, omega: np.array = None):
        """Construct a Zernike moments object

        :param omega: an optional moments matrix to initialise the class with
        """
        self._max_n = n_max

        if omega is not None:
            self._moments = omega
        else:
            self._moments = np.empty((n_max + 1, n_max + 1, n_max + 1), dtype=complex)

            # Fill with a number I will recognise if it's still left there
            # (which it shouldn't be for valid indexes)
            self._moments.fill(float('nan'))

    @property
    def dtype(self):
        return complex

    @property
    def max_order(self) -> int:
        return self._max_n

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
        return np.array([self[indices] for indices in self.iter_indices(redundant=True)])

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

    def value_at(self, x: np.array, order: int = None) -> float:
        """Reconstruct the value at x from the moments

        :param x: the point to get the value at
        :param order: the maximum order to go up to (defaults to the order of these moments)
        """
        order = order or self._max_n
        return self.reconstruct(self.create_reconstruction_query(x, order), order, zero_outside_domain=False)

    def reconstruct(self, query: ZernikeReconstructionQuery, order=None, zero_outside_domain=True):
        order = order if order is not None else self._max_n
        moments = query.moments

        values = np.zeros(query.points.shape[0], dtype=self.dtype)

        if zero_outside_domain:
            # Use only reconstruction moments from within the domain
            moments = query.moments_in_domain
            calculated_values = np.zeros(query.valid_idxs.size, dtype=self.dtype)
        else:
            calculated_values = values

        for n, l, m in iter_indices(max_order=order):
            omega = self.moment(n, l, m)
            z = sum_chi_nlm(n, l, m, moments)
            calculated_values += omega * z

            if m != 0:
                # Now do the symmetric -m part
                z_conj = (-1)**m * z.conjugate()

                calculated_values += z_conj * self.moment(n, l, -m)

        if zero_outside_domain:
            np.put(values, query.valid_idxs, calculated_values)

        return values.real

    @classmethod
    def create_reconstruction_query(cls, points: np.ndarray, order=None) -> ZernikeReconstructionQuery:
        """Create a query object that can be passed to self.value_at to reconstruct the Zernike functions
        at the given query points.  This is useful if the same point (or grid of points) is used multiple
        times as this precomputation can be done once and reused.

        :param points: the point to get the value at
        :param order: the maximum order to go up to (defaults to the order of these moments)
        """
        return ZernikeReconstructionQuery(points, cls._get_geometric_moments(points, order))

    @classmethod
    def _get_geometric_moments(cls, point: np.array, order: int) -> np.ndarray:
        """Get geometric moments from a point or set of points.  Used in reconstruction."""
        if len(point.shape) == 2:
            moms = [geometric.from_deltas(order, [pt]).to_matrix() for pt in point]
            moments = np.empty(list(moms[0].shape) + [len(moms)])
            for idx, entry in enumerate(moms):
                moments[:, :, :, idx] = entry
        else:
            moments = geometric.from_deltas(order, [point])

        return moments

    @staticmethod
    def _get_indices_in_domain(points: np.ndarray) -> np.ndarray:
        """Calculate the lengths squared and get the corresponding indexes"""
        length_sq = (points**2).sum(axis=1)
        return np.argwhere(length_sq <= 1)


class ZernikeMomentCalculator(functions.Function):
    """Calculate Zernike moments.

    Takes as input geometric moments or any state vector that is a valid input to GeometricMomentsCalculator
    """

    output_type = ZernikeMoments
    supports_jacobian = True
    dtype = complex

    def __init__(self, max_order: int):
        super().__init__()
        self._max_order = max_order
        self._geometric_moments_calculator = geometric.GeometricMomentsCalculator(max_order)
        self._jacobian = None
        self._chi = None  # The omega_nl_m prefactors

    def evaluate(self,
                 state: functions.State,
                 get_jacobian=False) -> Union[ZernikeMoments, Tuple[ZernikeMoments, np.ndarray]]:
        moments = ZernikeMoments(self._max_order)
        if isinstance(state, geometric.GeometricMoments):
            geom_moments = state
        else:
            geom_moments = self._geometric_moments_calculator(state, get_jacobian)
        geom_jac = None

        if get_jacobian:
            geom_moments, geom_jac = geom_moments

        # Get the moments themselves from polynomials of geometric moments
        for (n, l, m), poly in self.chi.items():
            moments[n, l, m] = poly.evaluate(geom_moments.moments)

        if get_jacobian:
            jac = np.matmul(self._get_jacobian(), geom_jac)
            return moments, jac

        return moments

    def _get_jacobian(self):
        if self._jacobian is None:
            # Calculate the first time
            self._jacobian = get_jacobian_wrt_geom_moments(self._max_order)

        return self._jacobian

    @property
    def chi(self) -> Dict[base_moments.Index, base_moments.MomentsPolynomial]:
        """Get the chi_nlm dictionary.  Indices are moment indices and values are polynomials"""
        if self._chi is None:

            class Tracker:
                """Keeps track of the moments evaluated by omega_nl_m"""

                def __init__(self, max_order):
                    self._max_order = max_order

                def __getitem__(self, item):
                    mtx = np.zeros((self._max_order + 1, self._max_order + 1, self._max_order + 1), dtype=complex)
                    mtx[item] = 1.
                    return mtx

            chi = collections.OrderedDict()
            for n, l, m in iter_indices(self._max_order, redundant=False):
                used = omega_nl_m(n, l, m, Tracker(self._max_order))
                indexes = np.argwhere(used)
                poly = base_moments.MomentsPolynomial()

                for index in indexes:
                    index = tuple(index)
                    poly.append(base_moments.Product(used[index], (index,)))

                chi[(n, l, m)] = poly

            self._chi = chi

        return self._chi


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


def sum_chi_nlm(n: int, l: int, m: int, geom_moments: np.ndarray) -> Union[complex, np.array]:
    """Calculate the Zernike geometric moment conversion factors"""
    return c_l_m(l, m) * 2**(-m) * sum1(n=n, l=l, m=m, geom_moments=geom_moments)


def sum1(n: int, l: int, m: int, geom_moments: np.ndarray) -> Union[complex, np.array]:
    """Calculate the Zernike geometric moment conversion factors"""
    k = int((n - l) / 2)  # n - l is always even
    total = 0.
    for nu in inclusive(k):
        total += q_kl_nu(k, l, nu) * sum2(n=n, l=l, m=m, nu=nu, geom_moments=geom_moments)

    return total


def sum2(n: int, l: int, m: int, nu: int, geom_moments: np.ndarray) -> Union[complex, np.array]:
    total = 0.
    for alpha in inclusive(nu):
        total += binomial_coeff(nu, alpha) * \
                 sum3(n=n, l=l, m=m, nu=nu, alpha=alpha, geom_moments=geom_moments)
    return total


def sum3(n: int, l: int, m: int, nu: int, alpha: int, geom_moments: np.ndarray) -> Union[complex, np.array]:
    total = 0.
    for beta in inclusive(nu - alpha):
        total += binomial_coeff(nu - alpha, beta) * \
                 sum4(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, geom_moments=geom_moments)
    return total


def sum4(n: int, l: int, m: int, nu: int, alpha: int, beta: int, geom_moments: np.ndarray) -> Union[complex, np.array]:
    total = 0.
    for u in inclusive(m):
        total += (-1) ** (m - u) * binomial_coeff(m, u) * 1j ** u * \
                 sum5(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u,
                      geom_moments=geom_moments)
    return total


def sum5(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int,
         geom_moments: np.ndarray) -> Union[complex, np.array]:
    total = 0.
    for mu in inclusive(int((l - m) / 2.)):
        total += (-1) ** mu * 2 ** (-2 * mu) * binomial_coeff(l, mu) * \
                 binomial_coeff(l - mu, m + mu) * \
                 sum6(n=n, l=l, m=m, nu=nu, alpha=alpha, beta=beta, u=u, mu=mu,
                      geom_moments=geom_moments)
    return total


def sum6(n: int, l: int, m: int, nu: int, alpha: int, beta: int, u: int, mu: int,
         geom_moments: np.ndarray) -> Union[complex, np.array]:
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
        raise ValueError(f'm must be in the range -{l} <= m <= {l}, got {m}')
    if not even(n - l):
        raise ValueError('n - l must be even')


def omega_nl_m(n: int, l: int, m: int, geom_moments: np.array) -> complex:
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


def _domain_check(positions: np.array):
    """Check that a given set of positions are within the domain for which Zernike moments are
    defined i.e. that |r| <= 1."""
    for idx, pos in enumerate(positions):
        if np.dot(pos, pos) > 1.:
            _LOGGER.warning(
                'Delta function %i is outside of the domain of Zernike basis functions '
                'with are defined within the unit ball', idx
            )
