# -*- coding: utf-8 -*-
import collections
from typing import List, Sequence, Sized, Optional, Union, Tuple

import ase
import matplotlib.pyplot as plt
import numpy as np

__all__ = 'SystemInfo', 'FingerprintSet', 'create_fingerprint_set'

SystemInfo = collections.namedtuple('SystemInfo', 'atoms fingerprints derivatives')


class FingerprintSet:
    """Container to store and analyse a set of fingerprints"""

    def __init__(self, fingerprint_length: int):
        self._fingerprint_length = fingerprint_length
        self._system_info: List[SystemInfo] = []

    def __len__(self) -> int:
        """Return the number of systems (not, this is NOT the number of environments)"""
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

    @property
    def systems(self) -> List[ase.Atoms]:
        return [info.atoms for info in self._system_info]

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
        if len(system) == 0:
            raise ValueError('Cannot add an empty atoms object')

        if len(system) != len(fingerprints):
            raise ValueError(
                'There must be as many fingerprints as there are atoms, ' \
                'got {} atoms and {} environments'.format(len(system), len(fingerprints)))
        if derivatives is not None and len(derivatives) != len(system):
            raise ValueError('Number of derivatives must match the number of atoms (environments) in the system')

        derivatives = np.array(derivatives) if derivatives is not None else None
        self._system_info.append(SystemInfo(system, np.array(fingerprints), derivatives))

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


def create_fingerprint_set(
    descriptor: 'milad.Descriptor', systems: Sequence[ase.Atoms], get_derivatives=False
) -> FingerprintSet:
    """Given a descriptor and a sequence of ase.Atoms objects this will create a fignerprint set"""
    # pylint: disable=too-many-locals
    from milad.play import asetools

    # WARNING: The calculation of derivatives only takes into account positional degrees of freedom (not species) and
    # makes assumptions about the shape of the derivatives tensor
    fp_length = descriptor.fingerprint_len

    fingerprints = FingerprintSet(descriptor.fingerprint_len)
    for system in systems:
        fps = []
        natoms = len(system)
        derivs = np.zeros((natoms, natoms, 3, fp_length)) if get_derivatives else None

        for my_idx, env in asetools.extract_environments(
            system, cutoff=descriptor.cutoff, yield_indices=True, include_central_atom=False
        ):
            milad_env = asetools.ase2milad(env)
            if get_derivatives:
                fingerprint, jacobian = descriptor(milad_env, jacobian=True)

                # The yielded environment has this array that allows us to map back on to the index in the original
                # structure
                orig_indices = env.get_array('orig_indices', copy=False)
                for i in range(len(env)):
                    neighbour_idx = orig_indices[i]
                    # Add the derivatives as the same neighbour may contribute more than once
                    derivs[my_idx, neighbour_idx, :, :] += jacobian[:, i * 3:(i + 1) * 3].T  # pylint: disable=unsupported-assignment-operation

                fps.append(fingerprint)
            else:
                fps.append(descriptor(milad_env))

        fingerprints.add_system(system, fps, derivatives=derivs)

    return fingerprints
