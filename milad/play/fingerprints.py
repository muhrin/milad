# -*- coding: utf-8 -*-
from typing import Iterable

import numpy

import milad
from . import envs

__all__ = ("FingerprintCalculator",)


class FingerprintCalculator:
    def __init__(
        self,
        invariants=None,
        sigma=0.5,
        cutoff: float = 6.0,
        cutoff_function="cos",
        normalise=True,
    ):
        self._invariants = invariants or milad.invariants.read_invariants(read_max=10)
        self._sigma = sigma
        self._cutoff = cutoff
        self._cutoff_function = cutoff_function
        self._normalise = normalise

    @property
    def cutoff(self) -> float:
        return self._cutoff

    def calculate(self, positions: numpy.array) -> numpy.array:
        num = len(positions)
        invs_size = len(self._invariants)

        fingerprint = numpy.empty((num, invs_size))

        for i, pos in enumerate(positions):
            env = envs.SmoothGaussianEnvironment(
                pos, self._cutoff, self._cutoff_function
            )
            for other in positions:
                env.add_gaussian(other, self._sigma)

            fingerprint[i] = env.calc_moment_invariants(
                self._invariants, normalise=self._normalise
            )

        return numpy.reshape(fingerprint, (num * invs_size))

    def calculate_neighbours(self, pos: numpy.array, neighbours: Iterable[numpy.array]):
        env = envs.SmoothGaussianEnvironment(pos, self._cutoff, self._cutoff_function)
        for neighbour in neighbours:
            env.add_gaussian(neighbour, self._sigma)

        return env.calc_moment_invariants(self._invariants, normalise=self._normalise)
