# -*- coding: utf-8 -*-
import collections
from typing import Union, Sized, Iterator, Sequence, Optional, Tuple, List

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
            dr = vectors[i] - vectors[j]  # pylint: disable=invalid-name
            lengths.append(np.linalg.norm(dr))

    if sort_result:
        lengths.sort()
    return lengths


def even(val: int) -> bool:
    """Test if an integer is event.  Returns True if so."""
    return (val % 2) == 0


def odd(val: int) -> bool:
    """Test if an integer is odd.  Returns True if so."""
    return not even(val)


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


SystemInfo = collections.namedtuple('SystemInfo', 'atoms fingerprints derivatives')


class FingerprintSet:
    """Container to store a set of fingerprints"""

    def __init__(self, fingerprint_length: int):
        self._fingerprint_length = fingerprint_length
        self._system_info: List[SystemInfo] = []

    def __len__(self):
        return len(self._system_info)

    @property
    def fingerprint_len(self) -> int:
        """Get the length of the fingerprint vector"""
        return self._fingerprint_length

    @property
    def total_environments(self) -> int:
        """Get the total number of environments summing up over all of the systems"""
        return sum(len(info.atoms) for info in self._system_info)

    @property
    def fingerprints(self) -> Sequence:
        """Access all the fingerprints"""
        return tuple(info.fingerprints for info in self._system_info)

    def has_all_derivatives(self):
        try:
            self.fingerprint_derivatives.index(None)
            return False
        except ValueError:
            # None not found
            return True

    @property
    def fingerprint_derivatives(self) -> Sequence:
        """Access all the fingerprint derivatives"""
        return tuple(info.derivatives for info in self._system_info)

    @property
    def sizes(self):
        return tuple(len(info.atoms) for info in self._system_info)

    def get_potential_energies(self, normalise=True) -> tuple:
        """Get the potential energies of all systems in the set"""
        if normalise:
            return tuple(info.atoms.get_potential_energy() / len(info.atoms) for info in self._system_info)

        return tuple(info.atoms.get_potential_energy() for info in self._system_info)

    def has_all_forces(self) -> bool:
        try:
            self.get_forces().index(None)
            return False
        except ValueError:
            # None not found
            return True

    def get_forces(self) -> Tuple[np.ndarray]:
        all_forces = []
        for info in self._system_info:
            try:
                forces = info.atoms.get_array('force')
            except KeyError:
                forces = info.atoms.get_forces()
            all_forces.append(forces)

        return tuple(all_forces)

    def add_system(self, system: ase.Atoms, fingerprints: Sized, derivatives: Optional[Sized] = None):
        """Add a system along with the fingerprints for all its environments"""
        if len(system) != len(fingerprints):
            raise ValueError(
                'There must be as many fingerprints as there are atoms, ' \
                'got {} atoms and {} environments'.format(len(system), len(fingerprints)))

        derivatives = np.array(derivatives) if derivatives is not None else None
        self._system_info.append(SystemInfo(system, fingerprints, derivatives))

    def systemwise_sum(self, values, normalise=True):
        """Given a vector of values, one per environment, this will sum up the values for each atomic system
        and return the result as a container whose length is the same as the number of systems

        The sum can optionally be normalised by the number of atoms in each system.
        """
        out = []
        idx = 0
        for info in self._system_info:
            natoms = len(info.atoms)
            summed = sum(values[idx:idx + natoms])
            if normalise:
                summed = summed / natoms
            out.append(summed)
            idx += natoms

        return out

    def plot_environments(self):
        """Create a plot of the environments.  Returns the matplotlib figure when can then be .show()n"""
        fig, axes = plt.subplots(figsize=(16, 5))

        for envs in self.fingerprints:
            for env in envs:
                axes.plot(tuple(range(len(env))), env, linewidth=1.0)

        return fig

    def plot_energies(self):
        fig, axes = plt.subplots(figsize=(16, 5))

        energies = self.get_potential_energies(normalise=True)
        axes.hist(energies, 50, log=True)
        axes.set_xlabel('Energy')
        axes.set_ylabel('Number')

        return fig

    def split(self, split_point: Union[float, int]) -> Tuple['FingerprintSet', 'FingerprintSet']:
        """Split this set into two.  This can be used for creating a training and validation set.

        The split point can be an integer, in which case it is treated as the index of the split point,
        or a float in which case is it treated as a ratio.
        """
        if isinstance(split_point, float):
            split_point = int(round(len(self) * split_point))
        elif not isinstance(split_point, int):
            raise TypeError('split_point must be integer or float, got {}'.format(split_point.__class__.__name__))

        # Split into two halves
        one = FingerprintSet(self.fingerprint_len)
        for info in self._system_info[:split_point]:
            one.add_system(*info)

        two = FingerprintSet(self.fingerprint_len)
        for info in self._system_info[split_point:]:
            two.add_system(*info)

        return one, two


def nl_pairs(     # pylint: disable=invalid-name
    n: Union[int, Tuple[int, int]],
    l: Union[int, Tuple[int, int]] = None,
    l_le_n=True,
    n_minus_l_even=True
) -> Iterator[Tuple]:
    """Generator that will create n,l pairs for spherical harmonics optionally with l <= s

    :param n: the maximum value of n to go up to
    :param l: the maximum value of l to go up to
    :param l_le_n: only yield pairs that satisfy l <= n
    :param n_minus_l_even: only yield pairs that satisfy even(l - n) == True
    """
    if not isinstance(n, tuple):
        n = (0, n)
    if l is None:
        l = 0, n[1]
    elif not isinstance(l, tuple):
        # have max l
        l = 0, l

    for n_ in inclusive(*n, 1):  # pylint: disable=invalid-name
        if n_minus_l_even and odd(l[0] - n_):
            l_start = l[0] + 1
        else:
            l_start = l[0]

        for l_ in inclusive(l_start, min(n_, l[1]) if l_le_n else l[1], 2 if n_minus_l_even else 1):  # pylint: disable=invalid-name
            yield n_, l_
