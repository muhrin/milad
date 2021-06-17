# -*- coding: utf-8 -*-
"""Module for calculating Zernike moments"""

import collections
import copy
import logging
import itertools
import math
import numbers
import functools
import random
from typing import Union, Tuple, Iterator, Dict, Optional

import numpy as np
from numpy import ma
import scipy
import scipy.special

from . import base_moments
from . import functions
from . import geometric
from . import sph
from .utils import even, inclusive

_LOGGER = logging.getLogger(__name__)

# pylint: disable=invalid-name

__all__ = 'ZernikeMoments', 'ZernikeMomentCalculator', 'from_deltas', 'from_gaussians'


def from_deltas(
    n_max: int,
    positions: np.array,
    weights: Union[numbers.Number, np.array] = 1.,
    l_max: int = None
) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a collection of delta functions with optional weights

    :param n_max: the maximum order of Zernike functions to use
    :param positions: the Cartesian positions of the delta functions
    :param weights: the weights of the delta functions, can be a scalar or np.array of the same
        length as positions
    :param l_max: the maximum spherical harmonic angular frequency, l, to go up to
    """
    _domain_check(positions)
    geom_moments = geometric.from_deltas(n_max, positions, weights)
    return from_geometric_moments(n_max, geom_moments, l_max)


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


def from_geometric_moments(
    n_max: int, geom_moments: Union[geometric.GeometricMoments, np.array], l_max: int = None
) -> 'ZernikeMoments':
    """Create a set of Zernike moments from a set of geometric moments

    :param n_max: the maximum order of Zernike functions to use
    :param geom_moments: the geometric moments
    :param l_max: the maximum spherical harmonic angular frequency, l, to go up to
    """
    return ZernikeMomentCalculator(n_max, l_max)(geom_moments)


class ZernikeReconstructionQuery(base_moments.ReconstructionQuery):
    """A query object for Zernike moments that allows for faster reconstructions by caching the reconstruction grid"""

    def __init__(self, max_order: int, points: np.ndarray, moments: np.ndarray):
        super().__init__(max_order, points)
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


#: Helper tuple for working with Zernike moment indexes
ZernikeIndex = collections.namedtuple('ZernikeIndex', 'n l m')


class ZernikeMoments(base_moments.Moments):
    """A container class for calculated Zernike moments"""

    @classmethod
    def from_indexed(cls, indexed, max_order: int, dtype=None) -> 'ZernikeMoments':
        dtype = dtype or type(indexed[0, 0, 0])
        moments = cls(max_order, dtype=dtype)
        for (n, l, m) in moments.iter_indices():
            moments._set_moment(n, l, m, indexed[n, l, m], set_minus_m=False)
        return moments

    @classmethod
    def from_vector(cls, n_max: int, vec: np.array, redundant=True) -> 'ZernikeMoments':
        moms = ZernikeMoments(n_max, dtype=vec.dtype)

        for (n, l, m), value in zip(iter_indices(redundant=redundant), vec):
            if n > n_max:
                break
            moms._set_moment(n, l, m, value, set_minus_m=not redundant)  # pylint: disable=protected-access

        return moms

    @classmethod
    def rand(cls, n_max: int, num_range=(-5., 5.)):
        moms = ZernikeMoments(n_max)
        for nlm in moms.iter_indices(redundant=False):
            mom = random.uniform(*num_range)
            if nlm.m != 0:
                # Only l != 0 moments have a complex part
                mom += +1j * random.uniform(*num_range)
            moms[nlm] = mom

        return moms

    def __init__(self, n_max: int, l_max=None, dtype=complex):
        """Construct a Zernike moments object
        """
        self._indices = index_traits(n_max, l_max)
        self._dtype = dtype

        moments = sph.create_array(self._indices, dtype=dtype)
        self._moments = moments
        # Fill it with NaN so that it's easy to spot any values that haven't been set but should be
        self._moments.fill(np.nan)

        # Create an index array that allows us to map from a multi-index like [5, 3, -1] to a linear index
        idx_array = ma.masked_array(np.empty(moments.shape, dtype=np.int), mask=moments.mask)
        idx_array[~moments.mask] = np.arange(moments.count())
        self._idx_array = idx_array

    def __eq__(self, other: 'ZernikeMoments'):
        """Test if another set of moments are equal to this one"""
        if not isinstance(other, ZernikeMoments):
            return False

        return np.all(self.vector == other.vector)

    def __add__(self, other: 'ZernikeMoments') -> 'ZernikeMoments':
        if not isinstance(other, ZernikeMoments):
            raise ValueError(f'Expected ZernikeMoments, got {type(other)}')

        moms = copy.deepcopy(self)
        moms.array.data[:] += other.array.data[:]

        return moms

    @property
    def dtype(self):
        return self._dtype

    @property
    def array(self) -> np.ndarray:
        return self._moments

    def to_matrix(self):
        return self._moments

    @property
    def max_order(self) -> int:
        return self._indices.n[1]

    @property
    def n_max(self) -> int:
        return self._indices.n[1]

    @property
    def l_max(self) -> int:
        return self._indices.l[1]

    def get_builder(self, mask: 'Optional[ZernikeMoments]' = None):
        return ZernikeMomentsBuilder(self.max_order, mask=mask)

    def get_mask(self) -> 'ZernikeMoments':
        moms = ZernikeMoments(self.n_max, self.l_max, dtype=object)
        moms.array.fill(None)
        return moms

    def randomise(self, num_range=(-5, 5), indices=(None, None, None)):
        for n, l, m in iter_indices_extended(indices, self.n_max, redundant=False):
            mom = random.uniform(*num_range)
            if m != 0:
                # Only l != 0 moments have a complex part
                mom += 1j * random.uniform(*num_range)
            self._set_moment(n, l, m, mom)

        return self

    @property
    def vector(self) -> np.ndarray:
        """Return this set of moments as a vector"""
        return self._moments.compressed()

    def __getitem__(self, item):
        return self.moment(*item)

    def __setitem__(self, key: Union[int, Tuple], value):
        self._set_moment(*key, value)

    def _set_moment(self, n: int, l: int, m: int, value, set_minus_m=True):
        """Set an individual moment.  This will automatically set the corresponding -m value"""
        if value is None:
            # Special case, usually happens if this is a mask
            self._moments[n, l, m] = None
            self._moments[n, l, -m] = None
        else:
            if m == 0:
                try:
                    if value.imag > 1e-9:
                        logging.warning(
                            'Trying to set value of moment %i,%i,%i to %d, '
                            'however these moment should always be real so discarding imaginary', n, l, m, value
                        )
                    # value = np.real(value)
                except AttributeError:
                    pass
            elif set_minus_m:
                # Set the -m counterpart directly
                self._moments[n, l, -m] = (-1)**m * value.conjugate()

            # This will be reached for all m values
            self._moments[n, l, m] = value

    def iter(self, redundant=True):
        """Iterate tuples containing the moment indices and value"""
        for indices in self.iter_indices(redundant=redundant):
            yield indices, self[indices]

    def iter_indices(self, redundant=True):
        yield from iter_indices(n_max=self.n_max, redundant=redundant)

    def linear_index(self, index: base_moments.Index) -> int:
        """Given a triple of Zernike moment indices this will return the corresponding linearly ordered index"""
        return self._idx_array[index]

    @staticmethod
    def num_moments(max_order: int, redundant=True) -> int:
        """Get the total number of Zernike moments up to the maximum order"""
        return sum(1 for _ in iter_indices(max_order, redundant=redundant))

    def moment(self, n: int, l: int, m: int) -> complex:
        """Get the n, l, m^th moment"""
        return self._moments[n, l, m]

    def value_at(self, x: np.array, order: int = None) -> float:
        """Reconstruct the value at x from the moments

        :param x: the point to get the value at
        :param order: the maximum order to go up to (defaults to the order of these moments)
        """
        order = order or self.n_max
        return self.reconstruct(self.create_reconstruction_query(x, order), order, zero_outside_domain=False)

    def reconstruct(self,
                    query: ZernikeReconstructionQuery,
                    order=None,
                    zero_outside_domain=True) -> Union[float, np.ndarray]:
        """Given a reconstruction query this will return the values of the function at the specified grid values"""
        order = order if order is not None else query.max_order
        moments = query.moments

        if len(query.points.shape) > 1:
            values = np.zeros(query.points.shape[0], dtype=self.dtype)
        else:
            values = 0.

        if zero_outside_domain:
            # Use only reconstruction moments from within the domain
            moments = query.moments_in_domain
            calculated_values = np.zeros(query.valid_idxs.size, dtype=self.dtype)
        else:
            calculated_values = values

        for n, l, m in iter_indices(n_max=order, redundant=False):
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

    def visualise(
        self,
        viewer='plotly_widget',
        query: ZernikeReconstructionQuery = None,
        max_order=None,
        num_points=31,
        square=False
    ):
        # pylint: disable=too-many-locals

        if viewer.startswith('plotly_widget'):
            volume_plot = viewer.endswith('volume')

            # Only do imports here so some of these tools aren't needed for the rest of milad
            import ipywidgets
            import plotly.graph_objects as go
            from milad import plotlytools

            if query is None:
                grid_points = self.get_grid(num_points, restrict_to_domain=False)
                query = self.create_reconstruction_query(grid_points, self.max_order)
            else:
                grid_points = query.points

            grid_values = self.reconstruct(query, max_order, zero_outside_domain=True)
            if square:
                grid_values = grid_values**2

            default_isomin = (grid_values.max() - grid_values.min()) * 0.6 + grid_values.min()
            default_isomax = grid_values.max()

            if volume_plot:
                data = plotlytools.volume(
                    grid_points,
                    grid_values,
                    isomin=grid_values.mean(),
                    opacity=0.1,
                    autocolorscale=True,
                )
            else:
                data = plotlytools.isosurface(
                    grid_points,
                    grid_values,
                    isomin=grid_values.mean(),
                    opacity=0.1,
                    autocolorscale=True,
                )

            fig = go.Figure(data=[data])
            widget = go.FigureWidget(fig)

            default_opacity = 0.05 if volume_plot else 0.4

            # pylint: disable=unused-variable
            @ipywidgets.interact(
                opacity=(0., 1., 0.01),
                isomin=(grid_values.min(), grid_values.max(), 0.5),
                isomax=(grid_values.min(), grid_values.max(), 0.5)
            )
            def update(opacity=default_opacity, isomin=default_isomin, isomax=default_isomax):
                with widget.batch_update():
                    isosurface = widget.data[0]
                    isosurface.opacity = opacity
                    isosurface.isomin = isomin
                    isosurface.isomax = isomax

            return widget

        raise ValueError(f"Unknown viewer '{viewer}'")

    @classmethod
    def create_reconstruction_query(cls, points: np.ndarray, max_order: int) -> ZernikeReconstructionQuery:
        """Create a query object that can be passed to self.value_at to reconstruct the Zernike functions
        at the given query points.  This is useful if the same point (or grid of points) is used multiple
        times as this precomputation can be done once and reused.

        :param points: the point to get the value at
        :param max_order: the maximum order to go up to (defaults to the order of these moments)
        """
        return ZernikeReconstructionQuery(max_order, points, cls._get_geometric_moments(points, max_order))

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
        return np.argwhere(length_sq < 1)


class ZernikeMomentsBuilder(functions.Function):
    """This builder takes a vector of real values and constructs a set of complex Zernike moments."""
    input_type = np.ndarray
    supports_jacobian = True

    def __init__(self, n_max: int, l_max: int = None, mask: Optional[ZernikeMoments] = None):
        super().__init__()
        self._n_max = n_max
        self._l_max = n_max if l_max is None else l_max
        self._mask = mask
        self._indices = None

    def apply_mask(self, moms: ZernikeMoments):
        """Given a set of moments this will set any values present from the mask"""
        if self._mask is None:
            raise RuntimeError('Cannot apply mask as, mask is None')
        for zernike_index, value in self._mask.iter(redundant=False):
            if value is not None:
                moms[zernike_index] = value

    @property
    def inverse(self) -> 'ZernikeMomentsBuilder.Inverse':
        return ZernikeMomentsBuilder.Inverse(self)

    def evaluate(self, state: np.ndarray, *, get_jacobian=False):  # pylint: disable=arguments-differ
        indices_list = tuple(self.iter_indices())
        if len(state) != len(indices_list):
            raise ValueError(
                f'Got input vector of length {len(state)} but was expecting one of length {len(indices_list)}'
            )

        # Create the moments and optionally a Jacobian
        moms = ZernikeMoments(self._n_max)
        moms.array.fill(0.)
        if self._mask is not None:
            self.apply_mask(moms)
        jac = np.zeros((len(moms), len(indices_list)), dtype=complex) if get_jacobian else None

        # Let's create the complex moments
        for idx, (zernike_idx, num_type) in enumerate(indices_list):
            moms[zernike_idx] += num_type * state[idx]

            if get_jacobian:
                n, l, m = zernike_idx
                lin_idx = moms.linear_index(zernike_idx)
                jac[lin_idx, idx] += 1 * num_type
                if zernike_idx.m > 0:
                    lin_idx = moms.linear_index((n, l, -m))
                    jac[lin_idx, idx] += (-1)**m * (1 if num_type == 1. else -1j)

        if get_jacobian:
            return moms, jac

        return moms

    def iter_indices(self):
        # Skip all the ones that are masked
        if self._indices is None:
            # Lazily create the indices
            self._indices = tuple(self._iter_indices(self._n_max, self._l_max))

        for zernike_index, num_type in self._indices:
            if self._mask is None or self._mask[zernike_index] is None:
                yield zernike_index, num_type

    @classmethod
    def _iter_indices(cls, n_max: int, l_max: int):
        # First deal with m = 0 as these are all real
        indices = index_traits(n_max, l_max)

        for n in inclusive(0, n_max):
            for l in indices.iter_l(n):
                if not even(n - l):
                    continue

                yield ZernikeIndex(n, l, 0), 1.

        # Now deal with m > 0, these have both a real and imaginary part
        for num_type in (1., 1j):
            for n in inclusive(n_max):
                for l in indices.iter_l(n):
                    if not even(n - l):
                        continue

                    for m in inclusive(1, l, 1):
                        yield ZernikeIndex(n, l, m), num_type

    class Inverse(functions.Function):
        """Turn a set a Zernike moments into a vector"""
        supports_jacobian = False

        def __init__(self, forward: 'ZernikeMomentsBuilder'):
            super().__init__()
            self._forward = forward

        @property
        def inverse(self) -> 'ZernikeMomentsBuilder':
            return self._forward

        def evaluate(  # pylint: disable=unused-argument
                self, state: ZernikeMoments, *, get_jacobian=False) -> np.ndarray:
            moms = []
            for indices, num_type in self._forward.iter_indices():
                mom = state[indices]
                moms.append(mom.real if num_type == 1. else mom.imag)

            return np.array(moms)


class ZernikeMomentCalculator(base_moments.MomentsCalculator):
    """Calculate Zernike moments.

    Takes as input geometric moments or any state vector that is a valid input to GeometricMomentsCalculator
    """
    output_type = ZernikeMoments
    supports_jacobian = True
    dtype = complex
    input_type = geometric.GeometricMoments, np.ndarray, functions.Features

    def __init__(self, n_max: int, l_max: int = None):
        super().__init__()
        self._n_max = n_max
        self._l_max = l_max
        self._geometric_moments_calculator = geometric.GeometricMomentsCalculator(n_max)
        self._jacobian = None
        self._chi = None  # The omega_nl_m prefactors

    def evaluate(self,
                 state: functions.State,
                 *,
                 get_jacobian=False) -> Union[ZernikeMoments, Tuple[ZernikeMoments, np.ndarray]]:
        if isinstance(state, geometric.GeometricMoments):
            geom_moments = state
            geom_jac = np.eye(state.size)
        else:
            geom_moments = self._geometric_moments_calculator(state, get_jacobian)
            if get_jacobian:
                geom_moments, geom_jac = geom_moments

        moments = ZernikeMoments(self._n_max, self._l_max, dtype=object if geom_moments.dtype == object else complex)

        # Get the moments themselves from polynomials of geometric moments
        for (n, l, m), poly in self.chi.items():
            moments[n, l, m] = poly.evaluate(geom_moments)

        if get_jacobian:
            jac = np.matmul(self._get_jacobian(), geom_jac)
            return moments, jac

        return moments

    def _get_jacobian(self):
        if self._jacobian is None:
            # Calculate the first time
            self._jacobian = get_jacobian_wrt_geom_moments(self._n_max, redundant=True)

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
            for n, l, m in iter_indices(self._n_max, self._l_max, redundant=False):
                used = omega_nl_m(n, l, m, Tracker(self._n_max))
                indexes = np.argwhere(used)
                poly = base_moments.MomentsPolynomial()

                for index in indexes:
                    index = tuple(index)
                    poly.append(base_moments.Product(used[index], (index,)))

                chi[(n, l, m)] = poly

            self._chi = chi

        return self._chi

    def create_random(self, max_order: int = None) -> ZernikeMoments:
        max_order = max_order or self._n_max
        return ZernikeMoments.rand(max_order)


def rand(max_order: int) -> ZernikeMoments:
    return ZernikeMomentCalculator(max_order).create_random()


def get_jacobian_wrt_geom_moments(max_order: int, redundant=True):
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

    tracker = DerivativeTracker(max_order)
    num_moments = ZernikeMoments.num_moments(max_order, redundant=redundant)  # Number of zernike moments
    num_geometric_moments = geometric.GeometricMoments.num_moments(max_order)
    jacobian = np.zeros((num_moments, num_geometric_moments), dtype=complex)

    for idx, (n, l, m) in enumerate(iter_indices(max_order, redundant=redundant)):
        if m < 0:
            # These are taken care of during the positive m iteration
            continue

        vec = omega_nl_m(n, l, m, tracker)
        jacobian[idx, :] = vec

        if redundant and m != 0:
            minus_m = (-1)**m * vec.conjugate()
            minus_m_idx = linear_index((n, l, -m))
            jacobian[minus_m_idx, :] = minus_m

    return jacobian


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
    k = int(math.floor((n - l) / 2))  # n - l is always even
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
    for mu in inclusive(int(math.floor((l - m) / 2.))):
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


def iter_indices(n_max: int = None, l_max: int = None, *, redundant=False) -> Iterator[ZernikeIndex]:
    """Iterate over Zernike function indices in lexicographic order.

    If redundant is True then all valid indices will be generated including those where there is a
    symmetry relation e.g.: o_22^-2 = (-1)^2 o22^2.

    :param n_max: order to provide indices up to, if None will generate all indices
    :param l_max: the maximum angular frequency to go up to
    :param redundant: if True include redundant indices
    """
    upper = itertools.count() if n_max is None else inclusive(n_max)
    for n in upper:
        indices = index_traits(n, l_max if l_max is None else min(n, l_max))
        for l in indices.iter_l(n):
            for m in inclusive(0, l):
                yield ZernikeIndex(n, l, m)

            # negative m indices go AFTER positive because we rely on the wrapping-around of negative indices
            # to make these work they are actually at the end of the sequence of m values
            if redundant:
                for m in inclusive(-l, -1, 1):
                    yield ZernikeIndex(n, l, m)


def iter_indices_extended(key: Tuple[int, Tuple], max_order: int, redundant=False) -> Iterator[ZernikeIndex]:
    # pylint: disable=too-many-branches
    if isinstance(key, int):
        # Assume the caller is treating us like a vector
        n, l, m = triple_index(key)
    else:
        # Assume tuple
        n, l, m = key

    if n is None or l is None or m is None:
        # Define the range of n values
        if n is None:
            n_iter = inclusive(0 if l is None else l, max_order)
        else:
            n_iter = inclusive(n, n)

        for n_idx in n_iter:
            # Define the range of l values
            if l is None:
                l_iter = inclusive(0, n_idx)
            else:
                l_iter = inclusive(l, l)

            for l_idx in l_iter:
                if not even(n_idx - l_idx):
                    continue

                # Define the range of m values
                if m is None:
                    m_start = -l_idx if redundant else 0
                    m_iter = inclusive(m_start, l_idx)
                else:
                    m_iter = inclusive(m, m)

                for m_idx in m_iter:
                    yield ZernikeIndex(n_idx, l_idx, m_idx)

    else:
        # All direct indices
        yield ZernikeIndex(n, l, m)


def triple_index(linear: int, redundant=True) -> ZernikeIndex:
    """Get the triple index from the given lexicographic index"""
    entry = None
    for entry, _ in zip(iter_indices(redundant=redundant), range(linear + 1)):
        pass
    return entry  # pylint: disable=inconsistent-return-statements


@functools.lru_cache(maxsize=256)
def linear_index(index: base_moments.Index, redundant=True) -> int:
    """Given a triple of zernike function indices this will return a lexicographically ordered
    integer"""
    # pylint: disable=inconsistent-return-statements
    if not redundant and index[2] < 0:
        raise ValueError('m value cannot be negative')

    for linear, triple in enumerate(iter_indices(redundant=redundant)):
        if triple == index:
            return linear

    assert False, 'Should never reach here'


def _domain_check(positions: np.array):
    """Check that a given set of positions are within the domain for which Zernike moments are
    defined i.e. that |r| <= 1."""
    if positions.dtype == object:
        return

    for idx, pos in enumerate(positions):
        if np.dot(pos, pos) > 1.:
            _LOGGER.warning(
                'Delta function %i is outside of the domain of Zernike basis functions '
                'with are defined within the unit ball', idx
            )


def index_traits(n, l) -> sph.IndexTraits:
    return sph.IndexTraits(n, l, l_le_n=True, n_minus_l_even=True)
