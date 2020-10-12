# -*- coding: utf-8 -*-
"""Module for calculating geometric moments"""

import numbers
from typing import Union, Tuple

import numpy as np
import sympy

from . import base_moments
from . import generate
from . import utils

__all__ = 'gaussian_geometric_moments', 'from_gaussians', 'from_deltas'


class GeometricMoments(base_moments.Moments):

    @classmethod
    def empty(cls, max_order: int):
        return GeometricMoments(np.zeros(max_order + 1, max_order + 1, max_order + 1))

    def __init__(self, moments: np.array):
        self._moments = moments

    def __getitem__(self, index: Tuple):
        """Get the moment of the given index"""
        return self._moments[index]

    @property
    def dtype(self):
        return float

    def to_matrix(self) -> np.array:
        return self._moments

    def moment(self, p: int, q: int, r: int) -> float:
        return self._moments[p, q, r]

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

    moments = np.zeros((max_order + 1, max_order + 1, max_order + 1))
    for pos, sigma, weight in zip(positions, sigmas, weights):
        moments += gaussian_geometric_moments(max_order, pos, sigma, weight)

    return GeometricMoments(moments)


def from_deltas(
    max_order: int,
    positions: np.array,
    weights: Union[numbers.Number, np.array] = 1.,
    out_moments=None,
    pos_derivatives: np.array = None,
    weight_derivatives: np.array = None,
) -> Union[GeometricMoments, np.array]:
    """
    Calculate the geometric moments for a collection of delta functions at the given positions with
    the given weights.

    :param max_order: the maximum order to calculate moments up to
    :param positions: the positions of the delta functions
    :param weights: the weights of the delta functions
    :return: a max_order * max_order * max_order array of moments
    """
    try:
        dtype = positions.dtype
    except AttributeError:
        dtype = object

    ubound = max_order + 1  # Calculate to max order (inclusive)

    if out_moments is None:
        out_moments = np.zeros((ubound, ubound, ubound), dtype=dtype)
        geom_moms = GeometricMoments(out_moments)
    else:
        if isinstance(out_moments, GeometricMoments):
            geom_moms = out_moments
            out_moments = geom_moms.to_matrix()
        else:
            geom_moms = GeometricMoments(out_moments)

    num_points = positions.shape[0]
    weights = _to_array(weights, num_points)

    if pos_derivatives is not None:
        expected_shape = (ubound, ubound, ubound, num_points, 3)
        if pos_derivatives.shape != expected_shape:
            raise ValueError("Positional derivatives have the wrong shape, should be '{}'".format(expected_shape))

    if weight_derivatives is not None:
        expected_shape = (ubound, ubound, ubound, num_points)
        if weight_derivatives.shape != expected_shape:
            raise ValueError("Weight derivatives have the wrong shape, should be '{}'".format(expected_shape))

    # pylint: disable=invalid-name
    out_moments.fill(0.)
    moms = np.empty((ubound, 3), dtype=dtype)
    moms[0] = 1  # 0^th power always 1.
    for idx, (pos, weight) in enumerate(zip(positions, weights)):
        # Calculate each x, y, z raise to powers up to the upper bound
        for power in range(1, ubound):
            moms[power] = np.multiply(moms[power - 1, :], pos)

        this_moments = weight * utils.outer_product(moms[:, 0], moms[:, 1], moms[:, 2])
        out_moments += this_moments

        if pos_derivatives is not None:
            for p in range(ubound):
                for q in range(ubound):
                    for r in range(ubound):
                        # perform the derivative
                        pos_derivatives[p, q, r, idx, 0] = p * this_moments[p - 1, q, r]
                        pos_derivatives[p, q, r, idx, 1] = q * this_moments[p, q - 1, r]
                        pos_derivatives[p, q, r, idx, 2] = r * this_moments[p, q, r - 1]

        if weight_derivatives is not None:
            weight_derivatives[:, :, :, idx] = this_moments / weight

    return geom_moms


def from_deltas_analytic(max_order: int, num_particles: int, pos_symbols=None, weight_symbols=None):
    r = pos_symbols or sympy.IndexedBase('r')  # Positions of particles
    w = weight_symbols or sympy.IndexedBase('w')  # Weights of particles
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


def _deltas_from_vec(
    vec: np.array,
    find_points: bool,
    find_weights: bool,
    points: np.array,
    weights: np.array,
) -> np.array:
    num_points = len(points)
    if find_points:
        # The first set of numbers should be positions
        points = vec[:num_points * 3].rehsape((num_points, 3))

    if find_weights:
        # The last set of number should be the weights
        weights = vec[-num_points:]

    return points, weights


class DeltaReconstruction:

    def __init__(
        self,
        invariants,
        num_points: int,
        target: np.array,
        points: Union[int, np.array] = None,
        weights: Union[int, np.array] = None,
    ):
        self._invariants = invariants
        self._num_points = num_points
        self._target = target
        self._points = points
        self._weights = weights if weights is None else _to_array(weights, num_points)

        degrees_of_freedom = 0
        if self._points is None:
            # Have to find points
            degrees_of_freedom += num_points * 3
        if self._weights is None:
            # Have to find weights
            degrees_of_freedom += num_points
        self._degrees_of_freedom = degrees_of_freedom

        # Preallocate our arrays
        max_order = invariants.max_order
        ubound = max_order + 1

        self._moments = np.empty((ubound, ubound, ubound))
        self._invariant_values = np.empty(len(self._invariants))

        # Derivatives
        self._jacobian_mtx = np.zeros((len(self._invariants), self._degrees_of_freedom))
        self._pos_derivatives = np.empty((ubound, ubound, ubound, num_points, 3))
        self._weight_derivatives = np.empty((ubound, ubound, ubound, num_points))

    @property
    def degrees_of_freedom(self):
        return self._degrees_of_freedom

    def from_vec(
        self,
        vec: np.array,
    ) -> Tuple[np.array, np.array]:
        """Get delta function positions and weights from an optimisation vector"""
        num_points = self._num_points
        if self._points is None:
            # The first set of numbers should be positions
            points = vec[:num_points * 3].reshape((num_points, 3))
        else:
            points = self._points

        if self._weights is None:
            # The last set of number should be the weights
            weights = vec[-num_points:]
        else:
            weights = self._weights

        return points, weights

    def to_vec(self, positions=None, weights=None) -> np.array:
        vec = np.empty(self._degrees_of_freedom)
        if self._points is None:
            entries = self._num_points * 3
            vec[:entries] = positions.reshape(entries)
        if self._weights is None:
            vec[-self._num_points:] = weights
        return vec

    def residuals(self, vec: np.array, _target=None, callback=None) -> np.array:
        max_order = self._invariants.max_order
        current_pos, current_weights = self.from_vec(vec)

        current_moments = self._moments
        current_invariants = self._invariant_values

        # Calculate moments
        from_deltas(max_order, current_pos, current_weights, out_moments=current_moments)

        # Calculate invariants
        np.array(self._invariants.apply(current_moments, results=current_invariants, normalise=False))
        diff = current_invariants - self._target

        if callback is not None:
            callback(current_pos, current_weights, diff)

        # Return the residuals
        return diff

    def jacobian(self, vec: np.array, *_args, **_kwargs):
        max_order = self._invariants.max_order
        jacobian_mtx = self._jacobian_mtx
        current_moments = self._moments

        current_pos, current_weights = self.from_vec(vec)

        current_moments = from_deltas(
            max_order,
            current_pos,
            current_weights,
            out_moments=current_moments,
            pos_derivatives=self._pos_derivatives if self._points is None else None,
            weight_derivatives=self._weight_derivatives if self._weights is None else None,
        )

        # Reset the jacobian
        jacobian_mtx.fill(0.)

        for k, inv in enumerate(self._invariants):
            for indices, derivative in inv.derivatives().items():
                # Evaluate the derivative at this set of moments
                d_phi_here = derivative.apply(current_moments)

                current_idx = 0

                if self._points is None:
                    dm = self._pos_derivatives[indices[0], indices[1], indices[2]]

                    for i, dm_dx_i in enumerate(dm):
                        start = current_idx + 3 * i
                        end = start + 3
                        jacobian_mtx[k, start:end] += d_phi_here * dm_dx_i

                    current_idx += self._num_points * 3

                if self._weights is None:
                    dw = self._weight_derivatives[indices[0], indices[1], indices[2]]
                    jacobian_mtx[k, current_idx:] += d_phi_here * dw

        return jacobian_mtx


def gaussian_moments(
    order: int,
    mu: Union[numbers.Number, np.array],
    sigmas: Union[numbers.Number, np.array] = 0.4,
    weight: numbers.Number = 1.
) -> numbers.Number:
    """Get the nt^h moment of a n-dim Gaussian (or normal distribution) centred at `mu`
    with a standard deviation of `sigma`.

    Taken from:
    https://en.wikipedia.org/wiki/Normal_distribution#Moments
    Can be generalised to any order using confluent hypergeometric functions of the second kind.

    Another useful reference is:
    http://www.randomservices.org/random/special/Normal.html

    :param mu: the mean of the distribution
    :param sigmas: the standard deviation of the distribution
    :param order: the order of the moment to get
    :param weight: the total probability or mass of the normal distribution.  This is the zero^th
        moment be definition
    """
    if order > 16:
        raise NotImplementedError("Asked for order '{}', only up to order 16 implemented!".format(order))

    mu = np.array(mu)
    shape = mu.shape[0]
    sigmas = _to_array(sigmas, shape)

    if order == 0:
        mom = 1.
    if order == 1:
        mom = mu
    if order == 2:
        mom = mu ** 2 + \
              sigmas ** 2
    if order == 3:
        mom = mu ** 3 + \
              3 * mu * sigmas ** 2
    if order == 4:
        mom = mu ** 4 + \
              6 * mu ** 2 * sigmas ** 2 + \
              3 * sigmas ** 4
    if order == 5:
        mom = mu ** 5 + \
              10 * mu ** 3 * sigmas ** 2 + \
              5 * mu * 3 * sigmas ** 4
    if order == 6:
        mom = mu ** 6 + \
              15 * mu ** 4 * sigmas ** 2 + \
              15 * mu ** 2 * 3 * sigmas ** 4 + \
              15 * sigmas ** 6
    if order == 7:
        mom = mu ** 7 + \
              21 * mu ** 5 * sigmas ** 2 + \
              35 * mu ** 3 * 3 * sigmas ** 4 + \
              7 * mu * 15 * sigmas ** 6
    if order == 8:
        mom = mu ** 8 + \
              28 * mu ** 6 * sigmas ** 2 + \
              70 * mu ** 4 * 3 * sigmas ** 4 + \
              28 * mu ** 2 * 15 * sigmas ** 6 + \
              105 * sigmas ** 8
    if order == 9:
        mom = mu ** 9 + \
              36 * mu ** 7 * sigmas ** 2 + \
              126 * mu ** 5 * 3 * sigmas ** 4 + \
              84 * mu ** 3 * 15 * sigmas ** 6 + \
              9 * mu * 105 * sigmas ** 8
    if order == 10:
        mom = mu ** 10 + \
              45 * mu ** 8 * sigmas ** 2 + \
              210 * mu ** 6 * 3 * sigmas ** 4 + \
              210 * mu ** 4 * 15 * sigmas ** 6 + \
              45 * mu ** 2 * 105 * mu * sigmas ** 8 + \
              945 * sigmas ** 10
    if order == 11:
        mom = mu ** 11 + \
              55 * mu ** 9 * sigmas ** 2 + \
              330 * mu ** 7 * 3 * sigmas ** 4 + \
              462 * mu ** 5 * 15 * sigmas ** 6 + \
              165 * mu ** 3 * 105 * sigmas ** 8 + \
              11 * mu * sigmas ** 10
    if order == 12:
        mom = mu ** 12 + \
              66 * mu ** 10 * sigmas ** 2 + \
              495 * mu ** 8 * 3 * sigmas ** 4 + \
              924 * mu ** 6 * 15 * sigmas ** 6 + \
              495 * mu ** 4 * 105 * sigmas ** 8 + \
              66 * mu ** 2 * 945 * sigmas ** 10 + \
              10395 * sigmas ** 12
    if order == 13:
        mom = mu ** 13 + \
              78 * mu ** 11 * sigmas ** 2 + \
              715 * mu ** 9 * 3 * sigmas ** 4 + \
              1716 * mu ** 7 * 15 * sigmas ** 6 + \
              1287 * mu ** 5 * 105 * sigmas ** 8 + \
              286 * mu ** 3 * 945 * sigmas ** 10 + \
              13 * mu * 10395 * sigmas ** 12
    if order == 14:
        mom = mu ** 14 + \
              91 * mu ** 12 * sigmas ** 2 + \
              1001 * mu ** 10 * 3 * sigmas ** 4 + \
              3003 * mu ** 8 * 15 * sigmas ** 6 + \
              3003 * mu ** 6 * 105 * sigmas ** 8 + \
              1001 * mu ** 4 * 945 * sigmas ** 10 + \
              91 * mu ** 2 * 10395 * sigmas ** 12 + \
              135135 * sigmas ** 14
    if order == 15:
        mom = mu ** 15 + \
              105 * mu ** 13 * sigmas ** 2 + \
              1365 * mu ** 11 * 3 * sigmas ** 4 + \
              5005 * mu ** 9 * 15 * sigmas ** 6 + \
              6435 * mu ** 7 * 105 * sigmas ** 8 + \
              3003 * mu ** 5 * 945 * sigmas ** 10 + \
              455 * mu ** 3 * 10395 * sigmas ** 12 + \
              15 * mu * 135135 * sigmas ** 14
    if order == 16:
        mom = mu ** 16 + \
              120 * mu ** 14 * sigmas ** 2 + \
              1820 * mu ** 12 * 3 * sigmas ** 4 + \
              8008 * mu ** 10 * 15 * sigmas ** 6 + \
              12870 * mu ** 8 * 105 * sigmas ** 8 + \
              8008 * mu ** 6 * 945 * sigmas ** 10 + \
              1820 * mu ** 4 * 10395 * sigmas ** 12 + \
              120 * mu ** 2 * 135135 * sigmas ** 14 + \
              2027025 * sigmas ** 16

    return weight * mom


def gaussian_geometric_moments(max_order: int, mu: np.array, sigma: numbers.Number, weight: numbers.Number) -> np.array:
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
        moments[:, order] = gaussian_moments(order, mu, sigma, weight=1.0)

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
