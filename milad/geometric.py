# -*- coding: utf-8 -*-
"""Module for calculating geometric moments"""
import collections
import functools
import numbers
from typing import Union, Tuple, Optional, Iterator

import numpy as np
from numpy import ma
import sympy

from . import base_moments
from . import functions
from . import utils

__all__ = 'GeometricMoments', 'GeometricMomentsCalculator'

# pylint: disable=invalid-name

# Some constants to make understanding indexing easier
X = 0
Y = 1
Z = 2

GeometricIndex = collections.namedtuple('GeometricIndex', 'p q r')


class GeometricMoments(base_moments.Moments):

    @staticmethod
    def num_moments(max_order: int) -> int:
        """Get the total number of moments up to the given maximum order"""
        return len(tuple(iter_indices(max_order)))

    @classmethod
    def from_vector(cls, vec: np.array, max_order: int) -> 'GeometricMoments':
        moms_mtx = np.zeros((max_order + 1, max_order + 1, max_order + 1))
        indices = np.array(list(iter_indices(max_order))).T
        linear = np.ravel_multi_index(indices, moms_mtx.shape)

        # Put the vector values into the moments matrix
        np.put(moms_mtx, linear, vec)

        return GeometricMoments(moms_mtx)

    def __init__(self, moments: np.array):
        # Just trying this for now, let's see if it's possible to not force self._moments to be a numpy array
        max_order = moments.shape[0] - 1
        # Mask everything and then unmask the ones we want
        mask = np.ones(moments.shape, dtype=np.int)
        idx_array = np.empty(moments.shape, dtype=np.int)
        for i, indices in enumerate(iter_indices(max_order)):
            mask[indices] = 0
            idx_array[indices] = i

        self._moments = ma.array(moments, mask=mask)
        self._idx_array = ma.array(idx_array, mask=mask)

    def __getitem__(self, index: base_moments.Index):
        """Get the moment of the given index"""
        return self._moments[index]

    def __setitem__(self, index: base_moments.Index, value):
        """Get the moment of the given index"""
        self._moments[index] = value

    @property
    def dtype(self) -> type:
        try:
            return self._moments.dtype
        except AttributeError:
            return type(self._moments[0, 0, 0])

    @property
    def max_order(self) -> int:
        return self._moments.shape[0] - 1

    @property
    def vector(self) -> np.array:
        return self._moments.compressed()

    @property
    def moments(self) -> np.array:
        return self._moments

    def to_matrix(self) -> np.array:
        return self._moments

    def moment(self, p: int, q: int, r: int) -> float:
        return self._moments[p, q, r]

    def linear_index(self, index: base_moments.Index) -> int:
        return self._idx_array[index]

    def iter_indices(self) -> Iterator[base_moments.Index]:
        """Iterate through the valid indices of these moments"""
        yield from iter_indices(self.max_order)

    def value_at(self, x: np.array, max_order: int = None):
        """Reconstruct the value at x from the moments

        :param x: the point to get the value at
        :param max_order: the maximum order to go up to (defaults to the maximum order of these
            moments)
        """
        shape = self._moments.shape
        value = 0.
        for p in range(shape[0]):
            for q in range(shape[1]):
                for r in range(shape[2]):
                    value += self._moments[p, q, r] * (x**(p, q, r)).prod(axis=-1)

        return value

    def get_mask(self) -> 'GeometricMoments':
        geom_moms = np.empty(self._moments.shape, dtype=object)
        geom_moms.fill(None)
        return GeometricMoments(geom_moms)

    def get_builder(self, mask=None) -> Optional[functions.Function]:
        return functions.FromVectorBuilder(GeometricMoments, kwargs=dict(max_order=self.max_order))


def linear_index(max_order: int, index: base_moments.Index) -> int:
    if np.sum(index) > max_order:
        raise ValueError(f'The index passed ({index}) is of higher than the max order ({max_order})')

    for i, indices in enumerate(iter_indices(max_order)):
        if indices == index:
            return i


def iter_indices(max_order: int) -> Iterator[base_moments.Index]:
    for i in utils.inclusive(max_order):
        for j in utils.inclusive(max_order - i):
            for k in utils.inclusive(max_order - i - j):
                yield GeometricIndex(i, j, k)


class GeometricMomentsCalculator(base_moments.MomentsCalculator):
    """Function that calculates geometric moments up to some maximum order"""

    supports_jacobian = False

    def __init__(self, max_order: int):
        super().__init__()
        self._max_order = max_order

    def output_length(self, _in_state: functions.State) -> int:
        return GeometricMoments.num_moments(self._max_order)

    def evaluate(self, features: functions.State, *, get_jacobian=False) -> \
            Union[GeometricMoments, Tuple[GeometricMoments, np.ndarray]]:
        result = geometric_moments(features, self._max_order, get_jacobian, dtype=features.vector.dtype)

        if get_jacobian:
            return GeometricMoments(result[0]), result[1]

        return GeometricMoments(result)

    def create_random(self, max_order: int = None):
        max_val = 5.
        max_order = max_order or self._max_order
        return GeometricMoments(max_val * (2 * np.random.rand(max_order, max_order, max_order) - 1))


@functools.singledispatch
def geometric_moments(state: functions.State, max_order: int, get_jacobian: bool, dtype) -> np.ndarray:
    raise TypeError("Don't know how to calculate geometric moments for type '{}'".format(type(state).__name__))


@geometric_moments.register(functions.WeightedDelta)
def _(delta: functions.WeightedDelta, max_order: int, get_jacobian: bool, dtype):
    order = max_order

    moms = np.empty((order + 1, 3), dtype=dtype)
    moms[0] = 1  # 0^th power always 1.
    moms[1] = delta.pos  # 1^st power always the position itself

    # Calculate each x, y, z raise to powers up to the upper bound
    for power in utils.inclusive(2, order, 1):
        moms[power] = np.multiply(moms[power - 1, :], delta.pos)

    moments = delta.weight * utils.outer_product(moms[:, 0], moms[:, 1], moms[:, 2])

    if get_jacobian:
        indices = tuple(iter_indices(max_order))
        jacobian = np.zeros((len(indices), delta.LENGTH), dtype=dtype)

        for i, (p, q, r) in enumerate(indices):
            jacobian[i, delta.X] = p * moments[p - 1, q, r]
            jacobian[i, delta.Y] = q * moments[p, q - 1, r]
            jacobian[i, delta.Z] = r * moments[p, q, r - 1]
            jacobian[i, delta.WEIGHT] = moments[p, q, r] / delta.weight

        return moments, jacobian

    return moments


@geometric_moments.register(functions.WeightedGaussian)
def _(gaussian: functions.WeightedGaussian, max_order: int, get_jacobian: bool, dtype):
    """Calculate the geometric moments for a 3D Gaussian"""
    O = max_order

    partial_moments = np.empty((O + 1, 3), dtype=dtype)
    partial_moments[0, :] = 1.0  # 0^th order, mass is multiplied in at end

    for order in utils.inclusive(0, O):
        partial_moments[order, :] = np.array(gaussian_moment[order](gaussian.pos, gaussian.sigma))

    moments = utils.outer_product(partial_moments[:, 0], partial_moments[:, 1], partial_moments[:, 2])

    if get_jacobian:
        dg = np.empty((
            O + 1,  # Order
            2,  # d/dmu, d/dsigma
            3,  # x, y, z
        ))

        for order in utils.inclusive(max_order):
            dg[order, :, :] = np.array(gaussian_moment_derivatives[order](gaussian.pos, gaussian.sigma))

        # Let's get the indices and create a Jacobian matrix
        indices = tuple(iter_indices(max_order))
        jacobian = np.zeros((len(indices), gaussian.LENGTH), dtype=dtype)

        for i, (p, q, r) in enumerate(indices):
            # Positions
            jacobian[i, gaussian.X] = gaussian.weight * dg[p, 0, 0] * partial_moments[q, 1] * partial_moments[r, 2]
            jacobian[i, gaussian.Y] = gaussian.weight * dg[q, 0, 1] * partial_moments[p, 0] * partial_moments[r, 2]
            jacobian[i, gaussian.Z] = gaussian.weight * dg[r, 0, 2] * partial_moments[p, 0] * partial_moments[q, 1]

            # Sigma
            jacobian[i, gaussian.SIGMA] = gaussian.weight * (
                    dg[p, 1, 0] * partial_moments[q, 1] * partial_moments[r, 2] + \
                    partial_moments[p, 0] * dg[q, 1, 1] * partial_moments[r, 2] + \
                    partial_moments[p, 0] * partial_moments[q, 1] * dg[r, 1, 2]
            )

        # Weights
        jacobian[:, gaussian.WEIGHT] = moments[:, :, :]

        return gaussian.weight * moments, jacobian

    return gaussian.weight * moments


@geometric_moments.register(functions.Features)
def _(environment: functions.Features, max_order: int, get_jacobian, dtype):
    """Calculate the geometric moments for a set of features"""
    O = max_order
    idx = 0

    moments = np.empty((O + 1, O + 1, O + 1), dtype=dtype)
    moments.fill(0.)

    jacobians = []
    for feature in environment.features:
        if get_jacobian:
            moms, jac = geometric_moments(feature, max_order, get_jacobian, dtype)
            jacobians.append(jac)
        else:
            moms = geometric_moments(feature, max_order, get_jacobian, dtype)

        moments += moms

        idx += len(feature)

    if get_jacobian:
        return moments, np.concatenate(jacobians, axis=1)

    return moments


def from_gaussians(
    max_order: int,
    positions: np.ndarray,
    sigmas: Union[numbers.Number, np.array] = 0.4,
    weights: Union[numbers.Number, np.array] = 1.
) -> GeometricMoments:
    """Calculate the geometric moments for a collection of Gaussians at the given positions with
    the passed parameters.

    :param positions: the positions of the Gaussians
    :param sigmas: the standard deviations
    :param max_order: the maximum order to calculate moments up to
    :param weights: the masses of the Gaussians (or probabilities)
    :return: a max_order * max_order * max_order array of moments
    """
    positions = np.array(positions)
    shape = positions.shape[0]
    sigmas = _to_array(sigmas, shape)
    weights = _to_array(weights, shape)

    features = functions.Features()
    for pos, sigma, weight in zip(positions, sigmas, weights):
        features.add(functions.WeightedGaussian(pos, sigma, weight))

    return GeometricMomentsCalculator(max_order)(features)


def from_deltas(
    max_order: int,
    positions: np.array,
    weights: Union[numbers.Number, np.array] = 1,
    get_jacobian=False,
) -> Union[GeometricMoments, np.array]:
    """
    Calculate the geometric moments for a collection of delta functions at the given positions with
    the given weights.

    :param max_order: the maximum order to calculate moments up to
    :param positions: the positions of the delta functions
    :param weights: the weights of the delta functions
    :param get_jacobian: return the Jacobian along with the geometric moments
    :return: a max_order * max_order * max_order array of moments
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)

    if not len(positions.shape) == 2 or positions.shape[1] != 3:
        raise ValueError(
            "Positions must be a list of vectors or a numpy array with shape [n, 3], not: '{}".format(positions.shape)
        )

    shape = positions.shape[0]
    weights = _to_array(weights, shape)

    features = functions.Features()
    for pos, weight in zip(positions, weights):
        features.add(functions.WeightedDelta(pos, weight))

    return GeometricMomentsCalculator(max_order)(features, jacobian=get_jacobian)


def from_deltas_analytic(max_order: int, num_particles: int, pos_symbols=None, weight_symbols=None):
    r = pos_symbols or sympy.IndexedBase('r')  # Positions of particles pylint: disable=invalid-name
    w = weight_symbols or sympy.IndexedBase('w')  # Weights of particles pylint: disable=invalid-name
    ubound = max_order + 1  # Upper bound

    # pylint: disable=invalid-name

    moments = sympy.MutableDenseNDimArray.zeros(ubound, ubound, ubound)

    for idx in range(num_particles):
        moms = sympy.MutableDenseNDimArray.zeros(ubound, 3)
        moms[0, :] = sympy.Array([1, 1, 1])  # 0^th power always 1.

        # Calculate each x, y, z raise to powers up to the upper bound
        for power in range(1, ubound):
            moms[power, 0] = moms[power - 1, 0] * r[idx, 0]
            moms[power, 1] = moms[power - 1, 1] * r[idx, 1]
            moms[power, 2] = moms[power - 1, 2] * r[idx, 2]

        moments += w[idx] * sympy.tensorproduct(moms[:, 0], moms[:, 1], moms[:, 2])

    return moments


def gaussian_moment_0(mu: float, sigma: float = 1.) -> float:  # pylint: disable=unused-argument
    return 1.


def gaussian_moment_0_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:  # pylint: disable=unused-argument
    return 0., 0.


def gaussian_moment_1(mu: float, sigma: float = 1.) -> float:  # pylint: disable=unused-argument
    return mu


def gaussian_moment_1_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:  # pylint: disable=unused-argument
    return 1., 0.


def gaussian_moment_2(mu: float, sigma: float = 1.) -> float:
    return mu**2 + sigma**2


def gaussian_moment_2_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    return 2 * mu, 2 * sigma


def gaussian_moment_3(mu: float, sigma: float = 1.) -> float:
    return mu**3 + 3 * mu * sigma**2


def gaussian_moment_3_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    return 3 * mu**2 + 3 * sigma**2, 6 * mu * sigma


def gaussian_moment_4(mu: float, sigma: float = 1.) -> Union[numbers.Number, np.ndarray]:
    return mu ** 4 + \
           6 * mu ** 2 * sigma ** 2 + \
           3 * sigma ** 4


def gaussian_moment_4_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    dmu = 4 * mu ** 3 + \
          12 * mu * sigma ** 2
    dsigma = 12 * mu ** 2 * sigma + \
             12 * sigma ** 3

    return dmu, dsigma


def gaussian_moment_5(mu: float, sigma: float = 1.):
    return mu ** 5 + \
           10 * mu ** 3 * sigma ** 2 + \
           5 * mu * 3 * sigma ** 4


def gaussian_moment_5_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    dmu = 5 * mu ** 4 + \
          30 * mu ** 2 * sigma ** 2 + \
          15 * sigma ** 4

    dsigma = 20 * mu ** 3 * sigma + \
             60 * mu * sigma ** 3

    return dmu, dsigma


def gaussian_moment_6(mu: float, sigma: float = 1.) -> float:
    mom = mu ** 6 + \
          15 * mu ** 4 * sigma ** 2 + \
          15 * mu ** 2 * 3 * sigma ** 4 + \
          15 * sigma ** 6

    return mom


def gaussian_moment_6_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    dmu = 6 * mu ** 5 + \
          60 * mu ** 3 * sigma ** 2 + \
          90 * mu * sigma ** 4
    dsigma = 30 * mu ** 4 * sigma + \
             180 * mu ** 2 * sigma ** 3 + \
             90 * sigma ** 5

    return dmu, dsigma


def gaussian_moment_7(mu: float, sigma: float = 1.) -> float:
    return mu ** 7 + \
           21 * mu ** 5 * sigma ** 2 + \
           35 * mu ** 3 * 3 * sigma ** 4 + \
           7 * mu * 15 * sigma ** 6


def gaussian_moment_7_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    dmu = 7 * mu ** 6 + \
          105 * mu ** 4 * sigma ** 2 + \
          315 * mu ** 2 * sigma ** 4 + \
          105 * sigma ** 6
    dsigma = 42 * mu ** 5 * sigma + \
             420 * mu ** 3 * sigma ** 3 + \
             630 * mu * sigma ** 5

    return dmu, dsigma


def gaussian_moment_8(mu: float, sigma: float = 1.) -> float:
    return mu ** 8 + \
           28 * mu ** 6 * sigma ** 2 + \
           70 * mu ** 4 * 3 * sigma ** 4 + \
           28 * mu ** 2 * 15 * sigma ** 6 + \
           105 * sigma ** 8


def gaussian_moment_8_derivative(mu: float, sigma: float = 1.) -> Tuple[float, float]:
    dmu = 8 * mu ** 7 + \
          168 * mu ** 5 * sigma ** 2 + \
          840 * mu ** 3 * sigma ** 4 + \
          840 * mu * sigma ** 6

    dsigma = 56 * mu ** 6 * sigma + \
             840 * mu ** 4 * sigma ** 3 + \
             2520 * mu ** 2 * sigma ** 5 + \
             840 * sigma ** 7

    return dmu, dsigma


def gaussian_moment_9(mu: float, sigma: float = 1.) -> float:
    return mu ** 9 + \
           36 * mu ** 7 * sigma ** 2 + \
           126 * mu ** 5 * 3 * sigma ** 4 + \
           84 * mu ** 3 * 15 * sigma ** 6 + \
           9 * mu * 105 * sigma ** 8


def gaussian_moment_10(mu: float, sigma: float = 1.) -> float:
    return mu ** 10 + \
           45 * mu ** 8 * sigma ** 2 + \
           210 * mu ** 6 * 3 * sigma ** 4 + \
           210 * mu ** 4 * 15 * sigma ** 6 + \
           45 * mu ** 2 * 105 * mu * sigma ** 8 + \
           945 * sigma ** 10


def gaussian_moment_11(mu: float, sigma: float = 1.) -> float:
    return mu ** 11 + \
           55 * mu ** 9 * sigma ** 2 + \
           330 * mu ** 7 * 3 * sigma ** 4 + \
           462 * mu ** 5 * 15 * sigma ** 6 + \
           165 * mu ** 3 * 105 * sigma ** 8 + \
           11 * mu * sigma ** 10


def gaussian_moment_12(mu: float, sigma: float = 1.) -> float:
    return mu ** 12 + \
           66 * mu ** 10 * sigma ** 2 + \
           495 * mu ** 8 * 3 * sigma ** 4 + \
           924 * mu ** 6 * 15 * sigma ** 6 + \
           495 * mu ** 4 * 105 * sigma ** 8 + \
           66 * mu ** 2 * 945 * sigma ** 10 + \
           10395 * sigma ** 12


def gaussian_moment_13(mu: float, sigma: float = 1.) -> float:
    return mu ** 13 + \
           78 * mu ** 11 * sigma ** 2 + \
           715 * mu ** 9 * 3 * sigma ** 4 + \
           1716 * mu ** 7 * 15 * sigma ** 6 + \
           1287 * mu ** 5 * 105 * sigma ** 8 + \
           286 * mu ** 3 * 945 * sigma ** 10 + \
           13 * mu * 10395 * sigma ** 12


def gaussian_moment_14(mu: float, sigma: float = 1.) -> float:
    return mu ** 14 + \
           91 * mu ** 12 * sigma ** 2 + \
           1001 * mu ** 10 * 3 * sigma ** 4 + \
           3003 * mu ** 8 * 15 * sigma ** 6 + \
           3003 * mu ** 6 * 105 * sigma ** 8 + \
           1001 * mu ** 4 * 945 * sigma ** 10 + \
           91 * mu ** 2 * 10395 * sigma ** 12 + \
           135135 * sigma ** 14


def gaussian_moment_15(mu: float, sigma: float = 1.) -> float:
    return mu ** 15 + \
           105 * mu ** 13 * sigma ** 2 + \
           1365 * mu ** 11 * 3 * sigma ** 4 + \
           5005 * mu ** 9 * 15 * sigma ** 6 + \
           6435 * mu ** 7 * 105 * sigma ** 8 + \
           3003 * mu ** 5 * 945 * sigma ** 10 + \
           455 * mu ** 3 * 10395 * sigma ** 12 + \
           15 * mu * 135135 * sigma ** 14


def gaussian_moment_16(mu: float, sigma: float = 1.) -> float:
    return mu ** 16 + \
           120 * mu ** 14 * sigma ** 2 + \
           1820 * mu ** 12 * 3 * sigma ** 4 + \
           8008 * mu ** 10 * 15 * sigma ** 6 + \
           12870 * mu ** 8 * 105 * sigma ** 8 + \
           8008 * mu ** 6 * 945 * sigma ** 10 + \
           1820 * mu ** 4 * 10395 * sigma ** 12 + \
           120 * mu ** 2 * 135135 * sigma ** 14 + \
           2027025 * sigma ** 16


"""Get the nt^h moment of a n-dim Gaussian (or normal distribution) centred at `mu`
with a standard deviation of `sigma`.

Taken from:
https://en.wikipedia.org/wiki/Normal_distribution#Moments
Can be generalised to any order using confluent hypergeometric functions of the second kind.

Another useful reference is:
http://www.randomservices.org/random/special/Normal.html

:param mu: the mean of the distribution
:param sigma: the standard deviation of the distribution
:param order: the order of the moment to get
:param weight: the total probability or mass of the normal distribution.  This is the zero^th
    moment be definition
"""
gaussian_moment = [
    np.vectorize(gaussian_moment_0),
    np.vectorize(gaussian_moment_1),
    np.vectorize(gaussian_moment_2),
    np.vectorize(gaussian_moment_3),
    np.vectorize(gaussian_moment_4),
    np.vectorize(gaussian_moment_5),
    np.vectorize(gaussian_moment_6),
    np.vectorize(gaussian_moment_7),
    np.vectorize(gaussian_moment_8),
    np.vectorize(gaussian_moment_9),
    np.vectorize(gaussian_moment_10),
    np.vectorize(gaussian_moment_11),
    np.vectorize(gaussian_moment_12),
    np.vectorize(gaussian_moment_13),
    np.vectorize(gaussian_moment_14),
    np.vectorize(gaussian_moment_15),
    np.vectorize(gaussian_moment_16),
]

gaussian_moment_derivatives = [
    np.vectorize(gaussian_moment_0_derivative),
    np.vectorize(gaussian_moment_1_derivative),
    np.vectorize(gaussian_moment_2_derivative),
    np.vectorize(gaussian_moment_3_derivative),
    np.vectorize(gaussian_moment_4_derivative),
    np.vectorize(gaussian_moment_5_derivative),
    np.vectorize(gaussian_moment_6_derivative),
    np.vectorize(gaussian_moment_7_derivative),
    np.vectorize(gaussian_moment_8_derivative),
]


def _check_gaussian_moments_input(
    mu: Union[numbers.Number, np.ndarray],
    sigma: Union[numbers.Number, np.ndarray],
    weight: numbers.Number,
):
    if isinstance(mu, np.ndarray):
        output_length = len(mu)
        input_length = output_length + 1  # Plus one for the weight
        if isinstance(sigma, np.ndarray):
            if isinstance(sigma, np.ndarray) and sigma.shape != mu.shape:
                raise ValueError(f'Mismatch between mu and sigma shapes: {mu.shape} != {sigma.shape}')
            input_length += len(sigma)
        else:
            input_length += 1  # Sigma scalar
    else:
        # Assume it's a scalar
        if isinstance(sigma, np.ndarray):
            raise TypeError('Scalar mu passed with vector sigma')

        input_length = 3
        output_length = 1

    if not isinstance(weight, numbers.Number):
        raise TypeError(f"Expecting weight to be a scalar, got '{weight.__class__.__name__}'")

    return input_length, output_length


def gaussian_geometric_moments(
    # pylint: disable=invalid-name
    max_order: int,
    mu: np.array,
    sigma: numbers.Number,
    weight: numbers.Number
) -> np.array:
    """
    Get the geometric moments for a 3D Gaussian

    :param mu: the position of the Gaussian distribution
    :param sigma: the standard deviation (scalar - same in all directions)
    :param max_order: the maximum order to calculate moments for
    :param weight: the total mass of the Gaussian (or equivalently total probability)
    :return: the 3d moment tensor
    """
    ubound = max_order + 1  # Calculate to max order (inclusive)

    moments = np.zeros((3, ubound))
    moments[:, 0] = 1.0  # 0^th order, mass is multiplied in at end

    for order in range(1, ubound):
        # Use weight 1 for now and then multiply later
        moments[:, order] = gaussian_moment[order](mu, sigma)

    moments_3d = np.empty((ubound,) * 3)
    # pylint: disable=invalid-name

    for p in range(ubound):
        for q in range(ubound):
            for r in range(ubound):
                moments_3d[p, q, r] = moments[0, p] * moments[1, q] * moments[2, r]

    moments_3d *= weight
    return moments_3d


def _to_array(value: Union[numbers.Number, np.array], shape):
    if isinstance(value, numbers.Number):
        sarray = np.empty(shape)
        sarray.fill(value)
        return sarray

    return value
