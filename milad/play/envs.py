# -*- coding: utf-8 -*-
import abc
import functools
from typing import List, Callable, Union, Sequence

import numpy

import milad

__all__ = (
    'CutoffFunction', 'Distribution', 'Gaussian3D', 'GaussianEnvironment', 'SmoothGaussianEnvironment', 'cos_cutoff',
    'make_cutoff_params'
)

CutoffFunction = Callable[[float], float]


def cos_cutoff(cutoff: float, x: float):
    return 0.5 * (numpy.cos(numpy.pi * x / cutoff) + 1)


def make_cutoff_params(cutoff_fn: Callable[[float, float], float], cutoff: float):
    return {'cutoff': cutoff, 'cutoff_function': functools.partial(cutoff_fn, cutoff)}


def make_cutoff_function(name: str, cutoff: float = 6.):
    if name == 'cos':
        return functools.partial(cos_cutoff, cutoff)

    raise ValueError('Unknown cutoff function: {}'.format(name))


class Distribution(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def moment_tensor(self, max_order: int, normalise=False) -> numpy.array:
        """Calculate the moment tensor up to the given maximum order, optionally normalising.
        This will result in a numpy.array which has dimensions max_order * max_order * max_order"""

    def calc_moment_invariants(
        self, invariants: Sequence[milad.invariants.MomentInvariant], normalise=True
    ) -> numpy.array:
        """Given a sequence of invariants, calculate their values using the moments of this
        distribution"""
        # Figure out the maximum order of moment we will need for this set of invariants
        max_order = 0
        for inv in invariants:
            max_order = max(max_order, inv.max_order)

        raw_moments = self.moment_tensor(max_order, normalise)
        return numpy.fromiter((invariant.apply(raw_moments, normalise=normalise) for invariant in invariants),
                              dtype=numpy.float64)


class Gaussian3D(Distribution):
    """A simple 3D gaussian"""

    def __init__(self, pos: numpy.array, sigma: float, volume=1.):
        self.pos = pos
        self.sigma = sigma
        self.volume = volume

    def moment_tensor(self, max_order: int, normalise=False) -> numpy.array:
        return milad.moments.gaussian_geometric_moments(
            max_order=max_order, mu=self.pos, sigma=self.sigma, weight=self.volume
        )


class GaussianEnvironment(Distribution):
    """An environment made up of Gaussians"""

    def __init__(self):
        self._gaussians: List[Gaussian3D] = []

    def append(self, gaussian: Gaussian3D):
        self._gaussians.append(gaussian)

    def moment_tensor(self, max_order: int, normalise=False) -> numpy.array:
        """Calculate the raw moments of the collection of Gaussians"""
        moms = numpy.zeros((max_order + 1, max_order + 1, max_order + 1))
        for gaussian in self._gaussians:
            moms += gaussian.moment_tensor(max_order, normalise=normalise)
        return moms


class SmoothGaussianEnvironment(Distribution):
    """An environment with some spatial position and a cutoff sphere around it.  There can,
    optionally be a cutoff function applied"""

    def __init__(
        self,
        pos: numpy.array = numpy.zeros(3),
        cutoff: float = 6.,
        cutoff_function: Union[CutoffFunction, str] = None
    ):
        self._pos = pos
        self._cutoff_sq = cutoff * cutoff
        if isinstance(cutoff_function, str):
            self._cutoff_function = make_cutoff_function(cutoff_function)
        else:
            self._cutoff_function = cutoff_function
        self._gaussians = GaussianEnvironment()

    def add_gaussian(self, pos: numpy.array, sigma: float, weight=1.) -> Union[bool, float]:
        dr = pos - self._pos
        dist_sq = numpy.square(dr).sum()
        if dist_sq > self._cutoff_sq:
            return False

        if self._cutoff_function:
            weight *= self._cutoff_function(dist_sq**0.5)

        self._gaussians.append(Gaussian3D(dr, sigma, weight))
        return weight

    def add_gaussians(self, positions: numpy.array, sigma: float, mass=1.) -> Sequence[Union[bool, float]]:
        results = []
        for position in positions:
            results.append(self.add_gaussian(position, sigma, mass))

        return results

    def moment_tensor(self, max_order: int, normalise=False) -> numpy.array:
        return self._gaussians.moment_tensor(max_order, normalise)
