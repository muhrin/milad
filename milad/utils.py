# -*- coding: utf-8 -*-
from typing import Union, Sized, Iterator, Sequence

import ase
import matplotlib.pyplot as plt
import numpy as np

__all__ = 'calculate_all_pair_distances', 'FingerprintSet'


def calculate_all_pair_distances(vectors, sort_result=True):
    """Calculate all pair distances between the given vectors"""
    num = len(vectors)
    lengths = []
    for i in range(num - 1):
        for j in range(i + 1, num):
            dr = vectors[i] - vectors[j]
            lengths.append(np.linalg.norm(dr))

    if sort_result:
        lengths.sort()
    return lengths


def even(val: int) -> bool:
    """Test if an integer is event.  Returns True if so."""
    return (val % 2) == 0


def inclusive(*args) -> Iterator[int]:
    """Like range() but inclusive of upper bound and automatically does iteration of ranges with a
    negative step e.g. 0, -4 will produce a range containing 0, -1, -2, -3, -4"""
    if len(args) not in (1, 2, 3):
        raise ValueError('Takes one or two args, got: {}'.format(args))

    if len(args) == 3:
        # Assume form is start, stop, step
        start, stop, step = args
    else:
        if len(args) == 1:
            start = 0
            stop = args[0]
        else:
            start = args[0]
            stop = args[1]

        step = 1 if start <= stop else -1

    sign = 1 if step > 0 else -1
    idx = start
    while sign * (stop - idx) >= 0:
        yield idx
        idx += step


class CoefficientCapture:

    class Capture:

        def __init__(self, mtx: np.array, idx):
            self._mtx = mtx
            self._idx = idx

        def __mul__(self, other):
            print('Coeff: {}, idx: {}'.format(other, self._idx))
            self._mtx[self._idx] = other
            return self

        def __rmul__(self, other):
            return self.__mul__(other)

        def __iadd__(self, other):
            self._mtx[self._idx] += other

        def __riadd__(self, other):
            return self.__iadd__(other)

    def __init__(self, shape: tuple):
        self._mtx = np.zeros(shape)

    def __getitem__(self, item):
        print('Capturing {}'.format(item))
        return self.Capture(self._mtx, item)

    @property
    def mtx(self) -> np.array:
        return self._mtx


def outer_product(*array) -> np.array:
    if not array:
        raise ValueError('No arrays supplied')

    product = array[0]
    for entry in array[1:]:
        product = np.tensordot(product, entry, axes=0)

    return product


class FingerprintSet:
    """Container to store a set of fingerprints"""

    def __init__(self, fingerprint_length: int, systems=None, fingerprints=None):
        self._fingerprint_length = fingerprint_length
        self._systems = []
        self._fingerprints = []
        if systems:
            for system, envs in zip(systems, fingerprints):
                self.add_system(system, envs)

    def __len__(self):
        return len(self._systems)

    @property
    def fingerprint_len(self) -> int:
        """Get the length of the fingerprint vector"""
        return self._fingerprint_length

    @property
    def total_environments(self) -> int:
        """Get the total number of environments summing up over all of the systems"""
        return sum(len(system) for system in self._systems)

    @property
    def fingerprints(self) -> Sequence:
        """Access the fingerprints"""
        return self._fingerprints

    def get_potential_energies(self, normalise=True) -> tuple:
        """Get the potential energies of all systems in the set"""
        if normalise:
            return tuple(system.get_potential_energy() / len(system) for system in self._systems)

        return tuple(system.get_potential_energy() for system in self._systems)

    def add_system(self, system: ase.Atoms, fingerprints: Sized):
        """Add a system along with the fingerprints for all its environments"""
        if len(system) != len(fingerprints):
            raise ValueError(
                'There must be as many fingerprints as there are atoms, ' \
                'got {} atoms and {} environments'.format(len(system), len(fingerprints)))

        self._systems.append(system)
        self._fingerprints.append(fingerprints)

    def systemwise_sum(self, values, normalise=True):
        """Given a vector of values, one per environment, this will sum up the values for each atomic system
        and return the result as a container whose length is the same as the number of systems

        The sum can optionally be normalised by the number of atoms in each system.
        """
        out = []
        idx = 0
        for system_idx, system in enumerate(self._systems):
            natoms = len(system)
            summed = sum(values[idx:idx + natoms])
            if normalise:
                summed = summed / natoms
            out.append(summed)
            idx += natoms

        return out

    def plot_environments(self):
        """Create a plot of the environments.  Returns the matplotlib figure when can then be .show()n"""
        fig, axes = plt.subplots(figsize=(16, 5))

        for envs in self._fingerprints:
            for env in envs:
                axes.plot(tuple(range(len(env))), env, linewidth=1.0)

        return fig

    def split(self, split_point: Union[float, int]):
        """Split this set into two.  This can be used for creating a training and validation set.

        The split point can be an integer, in which case it is treated as the index of the split point,
        or a float in which case is it treated as a ratio.
        """
        if isinstance(split_point, float):
            split_point = int(round(len(self) * split_point))
        elif not isinstance(split_point, int):
            raise TypeError('split_point must be integer or float, got {}'.format(split_point.__class__.__name__))

        a = FingerprintSet(self.fingerprint_len, self._systems[:split_point], self._fingerprints[:split_point])
        b = FingerprintSet(self.fingerprint_len, self._systems[split_point:], self._fingerprints[split_point:])

        return a, b
