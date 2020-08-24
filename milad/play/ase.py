# -*- coding: utf-8 -*-
from typing import Union, Any, Callable

import ase.neighborlist
import numpy

import dscribe.utils.geometry

import milad
from . import distances
from . import envs
from . import fingerprints


def calculate_fingerprint(atoms: ase.Atoms,
                          calculator: fingerprints.FingerprintCalculator):
    num_atoms = len(atoms)
    nlist = ase.neighborlist.NeighborList(cutoffs=[0.5 * calculator.cutoff] *
                                          num_atoms,
                                          self_interaction=False,
                                          bothways=True)
    nlist.update(atoms)

    fps = []

    for idx, pos in enumerate(atoms.positions):
        indices, offsets = nlist.get_neighbors(idx)

        positions = []
        for i, offset in zip(indices, offsets):
            positions.append(atoms.positions[i] +
                             numpy.dot(offset, atoms.get_cell()))

        # positions.append(pos)

        fps.append(calculator.calculate_neighbours(pos, positions))

    return numpy.array(fps)


def calculate_fingerprints_dscribe(
        atoms: ase.Atoms, calculator: fingerprints.FingerprintCalculator):
    positions = atoms.positions
    cutoff = calculator.cutoff
    cutoff_sq = cutoff * cutoff

    system = dscribe.utils.geometry.get_extended_system(
        atoms, cutoff, return_cell_indices=False)

    fps = []
    for i, pos in enumerate(positions):
        neighbours = []
        for j, neighbour in enumerate(system.positions):
            if i == j:
                continue

            dr = pos - neighbour
            dr_sq = numpy.dot(dr, dr)
            if dr_sq < cutoff_sq:
                neighbours.append(neighbour)

        fps.append(calculator.calculate_neighbours(pos, neighbours))

    return numpy.array(fps)


def _create_params_dict(species: list, value: Union[Any, dict]) -> dict:
    if isinstance(value, dict):
        return value

    return {specie: value for specie in species}


class MiladFingerprint:
    def __init__(self,
                 species: list,
                 invariants=None,
                 sigmas=0.4,
                 weights=1.0,
                 cutoffs: Union[float, dict] = 6.,
                 cutoff_fn=None,
                 normalise=True,
                 split_specie_pairs=False):
        """
        :param species: a list of the species to consider
        :param invariants:
        :param sigmas:
        :param weights:
        :param cutoffs:
        :param cutoff_fn:
        :param normalise:
        :param split_specie_pairs: return a fingerprint that is a matrix containing specie pairs
        """
        self._species = species
        self._invariants = invariants or milad.invariants.read_invariants()
        self._sigmas = _create_params_dict(species, sigmas)
        self._weights = _create_params_dict(species, weights)
        self._cutoffs = _create_params_dict(species, cutoffs)
        self._cutoff_fns = _create_params_dict(species, cutoff_fn)
        self._normalise = normalise
        self._split_species = split_specie_pairs

    def set_params(self,
                   specie: str,
                   sigma: float = None,
                   weight: float = None,
                   cutoff: float = None,
                   cutoff_fn: Union[float, Callable] = None):
        """Set parameters for a particular specie"""
        if sigma is not None:
            self._sigmas[specie] = sigma
        if weight is not None:
            self._weights[specie] = weight
        if cutoff is not None:
            self._cutoffs[specie] = cutoff
        if cutoff_fn is not None:
            self._cutoff_fns[specie] = cutoff_fn

    def create(self, system: ase.Atoms, *args, **kwargs):
        positions = system.positions
        symbols = system.symbols

        fingerprint = numpy.zeros((len(positions), len(self._invariants)))

        if any(system.get_pbc()):
            dist_calculator = distances.UnitCellDistanceCalculator(
                system.cell.array)
        else:
            dist_calculator = distances.AperiodicDistanceCalculator()

        # Calculate the fingerprints for each atom centre one at a time
        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol],
                                                 self._cutoff_fns[symbol])

            # Add ourselves
            env.add_gaussian(pos,
                             sigma=self._sigmas[symbol],
                             weight=self._weights[symbol])
            cutoff = self._cutoffs[symbol]

            # Visit all neighbours
            for j, neighbour_pos in enumerate(positions):
                neighbour_symbol = symbols[j]
                vecs = dist_calculator.get_vecs_between(pos,
                                                        neighbour_pos,
                                                        cutoff=cutoff,
                                                        self_interation=i != j)

                for vec in vecs:
                    env.add_gaussian(pos + vec,
                                     sigma=self._sigmas[neighbour_symbol],
                                     weight=self._weights[neighbour_symbol])

            fingerprint[i, :] = env.calc_moment_invariants(
                self._invariants, normalise=self._normalise)

        return fingerprint

    def create_old(self, system: ase.Atoms, *args, **kwargs):
        positions = system.positions
        symbols = system.symbols

        fingerprint = numpy.zeros((len(positions), len(self._invariants)))

        nlist = ase.neighborlist.NeighborList(cutoffs=tuple(
            map(lambda sym: 0.5 * self._cutoffs[sym], symbols)),
                                              self_interaction=False,
                                              bothways=True)
        nlist.update(system)

        # Calculate the fingerprints for each atom centre one at a time
        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol],
                                                 self._cutoff_fns[symbol])

            # Add ourselves
            env.add_gaussian(pos,
                             sigma=self._sigmas[symbol],
                             weight=self._weights[symbol])

            # Visit all neighbours
            indices, offsets = nlist.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                neighbour_pos = positions[j] + numpy.dot(
                    offset, system.get_cell())
                neighbour_symbol = symbols[j]
                env.add_gaussian(neighbour_pos,
                                 sigma=self._sigmas[neighbour_symbol],
                                 weight=self._weights[neighbour_symbol])

            fingerprint[i, :] = env.calc_moment_invariants(
                self._invariants, normalise=self._normalise)

        return fingerprint

    def create_single_old(self, system: ase.Atoms, index):
        positions = system.positions
        symbols = system.symbols

        # Calculate the fingerprints for each atom centre one at a time
        pos = positions[index]
        symbol = symbols[index]

        if self._split_species:
            neighbour_types = [[specie] for specie in self._species]
        else:
            neighbour_types = [self._species]

        n_types = len(neighbour_types)
        fingerprint = numpy.zeros((n_types, len(self._invariants)))

        # Create the environments
        environs = []

        for _ in range(n_types):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol],
                                                 self._cutoff_fns[symbol])
            # Add ourselves
            env.add_gaussian(pos,
                             sigma=self._sigmas[symbol],
                             weight=self._weights[symbol])
            environs.append(env)

        # Create the neighbour list
        nlist = ase.neighborlist.NeighborList(cutoffs=tuple(
            map(lambda sym: 0.5 * self._cutoffs[sym], symbols)),
                                              self_interaction=False,
                                              bothways=True)
        nlist.update(system)

        # Go through all the neighbours and add Gaussians on each
        indices, offsets = nlist.get_neighbors(index)
        for i, offset in zip(indices, offsets):
            neighbour_pos = positions[i] + numpy.dot(offset, system.get_cell())
            neighbour_symbol = symbols[i]

            for type_idx, types in enumerate(neighbour_types):
                if neighbour_symbol in types:
                    environs[type_idx].add_gaussian(
                        neighbour_pos,
                        sigma=self._sigmas[neighbour_symbol],
                        weight=self._weights[neighbour_symbol])

        # Finally compute all the fingerprints
        for type_idx, env in enumerate(environs):
            fingerprint[type_idx] = env.calc_moment_invariants(
                self._invariants, normalise=self._normalise)

        if len(fingerprint) == 1:
            return fingerprint[0]

        return fingerprint

    def create_single(self, system: ase.Atoms, index):
        positions = system.positions
        symbols = system.symbols

        # Calculate the fingerprints for each atom centre one at a time
        pos = positions[index]
        symbol = symbols[index]
        cutoff = self._cutoffs[symbol]

        if self._split_species:
            neighbour_types = [[specie] for specie in self._species]
        else:
            neighbour_types = [self._species]

        n_types = len(neighbour_types)
        fingerprint = numpy.zeros((n_types, len(self._invariants)))

        # Create the environments
        environs = []

        for _ in range(n_types):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol],
                                                 self._cutoff_fns[symbol])
            # Add ourselves
            env.add_gaussian(pos,
                             sigma=self._sigmas[symbol],
                             weight=self._weights[symbol])
            environs.append(env)

        if any(system.get_pbc()):
            dist_calculator = distances.UnitCellDistanceCalculator(
                system.cell.array)
        else:
            dist_calculator = distances.AperiodicDistanceCalculator()

        for i, neighbour_pos in enumerate(positions):
            neighbour_symbol = symbols[i]
            vecs = dist_calculator.get_vecs_between(pos,
                                                    neighbour_pos,
                                                    cutoff=cutoff,
                                                    self_interation=i != index)
            for type_idx, types in enumerate(neighbour_types):
                if neighbour_symbol in types:
                    for vec in vecs:
                        environs[type_idx].add_gaussian(
                            pos + vec,
                            sigma=self._sigmas[neighbour_symbol],
                            weight=self._weights[neighbour_symbol])

        # Finally compute all the fingerprints
        for type_idx, env in enumerate(environs):
            fingerprint[type_idx] = env.calc_moment_invariants(
                self._invariants, normalise=self._normalise)

        if len(fingerprint) == 1:
            return fingerprint[0]

        return fingerprint
