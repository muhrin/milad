# -*- coding: utf-8 -*-
import math

import numpy as np

from milad import functions, zernike, invariants

__all__ = 'Landscape', 'GenerativeLoss', 'Gaussian', 'RepulsiveForce'

ROOT_TWO_PI = (2 * math.pi)**0.5


class Gaussian(functions.Function):
    """An N-dimensional, isotropic, Gaussian function"""
    supports_jacobian = False

    def __init__(self, loc: np.ndarray, weight: float = 1., sigma: float = 1., debug=False):
        """
        :param loc: the location of the Gaussian
        :param weight: the weight
        :param sigma:
        """
        super().__init__()
        self._loc = loc
        self._loc_sum = np.sum(self._loc * self._loc)
        self._weight = weight
        self._sigma = sigma
        self._debug = debug

    @property
    def loc(self) -> np.ndarray:
        """Get the location of this Gaussian"""
        return self._loc

    def evaluate(
        # pylint: disable=unused-argument, arguments-differ
        self,
        pos: np.ndarray,
        *,
        get_jacobian=False
    ):
        dr = self.dist(pos)  # pylint: disable=invalid-name
        val = self._weight / (self._sigma * ROOT_TWO_PI) * np.exp(-(1 / 2) * (dr / self._sigma)**2)
        if self._debug:
            print('Dist: {}, Energy: {}'.format(dr, val))
        return val

    def dist(self, pos):
        """Get the distance to the given position"""
        dr = np.linalg.norm(pos - self._loc, ord=2)  # pylint: disable=invalid-name
        return dr

    @property
    def extremum(self) -> float:
        """Return the value at the turning point of the Gaussian.  This could be negative if it has negative weight"""
        return self._weight / (self._sigma * ROOT_TWO_PI)


class Landscape(functions.Function):
    supports_jacobian = False

    def __init__(self, *feature: Gaussian):
        super().__init__()
        self._features = list(feature)

    def add_feature(self, feature: Gaussian):
        """
        Add a feature to this landscape
        :param feature:
        :return:
        """
        self._features.append(feature)

    def evaluate(
        # pylint: disable=unused-argument, arguments-differ
        self,
        pos,
        *,
        get_jacobian=False
    ):
        minimum = min(feature.extremum for feature in self._features)
        # closest_idx = np.argmin([np.linalg.norm(feature.loc - pos) for feature in self._features])
        # energy = self._features[closest_idx](pos) - minimum
        energy = np.sum(feature(pos) - minimum for feature in self._features)
        return energy


class RepulsiveForce(functions.Function):
    input_type = zernike.ZernikeMoments
    output_type = float
    supports_jacobian = False

    def __init__(self, background: zernike.ZernikeMoments, strength: float = 1.):
        super().__init__()
        self.background = background
        self.strength = strength

    def evaluate(
        # pylint: disable=unused-argument, arguments-differ
        self,
        foreground: zernike.ZernikeMoments,
        *,
        get_jacobian=False
    ):
        force = np.abs(self.strength * np.sum(foreground.array * self.background.array.conj()).real)
        return force


class GenerativeLoss(functions.Function):
    supports_jacobian = False

    def __init__(self, landscape: Landscape, invs: invariants.MomentInvariants, repulsive_force: RepulsiveForce = None):
        super().__init__()
        self.landscape = landscape
        self.repulsive_forces = repulsive_force
        self._invs = invs

    def evaluate(
        # pylint: disable=unused-argument, arguments-differ
        self,
        moments: zernike.ZernikeMoments,
        *,
        get_jacobian=False
    ):
        repulsive = 0.

        if self.repulsive_forces is not None:
            # Repulsive part
            repulsive = self.repulsive_forces(moments)
            moments = moments + self.repulsive_forces.background

        # Energy landscape attraction
        fingerprint = self._invs(moments)
        # print(fingerprint)
        attractive = self.landscape(fingerprint)

        return attractive + repulsive
