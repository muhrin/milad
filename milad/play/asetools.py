# -*- coding: utf-8 -*-
from typing import Union, Any, Callable

import ase.neighborlist
import miniball
import numpy as np

import milad
from milad import atomic
from . import distances
from . import envs


def _create_params_dict(species: list, value: Union[Any, dict]) -> dict:
    if isinstance(value, dict):
        return value

    return {specie: value for specie in species}


class MiladFingerprint:

    def __init__(
        self,
        species: list,
        invariants=None,
        sigmas=0.4,
        weights=1.0,
        cutoffs: Union[float, dict] = 6.,
        cutoff_fn=None,
        normalise=True,
        split_specie_pairs=False
    ):
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

    def set_params(
        self,
        specie: str,
        sigma: float = None,
        weight: float = None,
        cutoff: float = None,
        cutoff_fn: Union[float, Callable] = None
    ):
        """Set parameters for a particular specie"""
        if sigma is not None:
            self._sigmas[specie] = sigma
        if weight is not None:
            self._weights[specie] = weight
        if cutoff is not None:
            self._cutoffs[specie] = cutoff
        if cutoff_fn is not None:
            self._cutoff_fns[specie] = cutoff_fn

    def create(self, system: ase.Atoms):
        # pylint: disable=too-many-locals
        positions = system.positions
        symbols = system.symbols

        fingerprint = np.zeros((len(positions), len(self._invariants)))

        if any(system.get_pbc()):
            dist_calculator = distances.UnitCellDistanceCalculator(system.cell.array)
        else:
            dist_calculator = distances.AperiodicDistanceCalculator()

        # Calculate the fingerprints for each atom centre one at a time
        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol], self._cutoff_fns[symbol])

            # Add ourselves
            env.add_gaussian(pos, sigma=self._sigmas[symbol], weight=self._weights[symbol])
            cutoff = self._cutoffs[symbol]

            # Visit all neighbours
            for j, neighbour_pos in enumerate(positions):
                neighbour_symbol = symbols[j]
                vecs = dist_calculator.get_vecs_between(pos, neighbour_pos, cutoff=cutoff, self_interation=i != j)

                for vec in vecs:
                    env.add_gaussian(
                        pos + vec, sigma=self._sigmas[neighbour_symbol], weight=self._weights[neighbour_symbol]
                    )

            fingerprint[i, :] = env.calc_moment_invariants(self._invariants, normalise=self._normalise)

        return fingerprint

    def create_old(self, system: ase.Atoms):
        # pylint: disable=too-many-locals

        positions = system.positions
        symbols = system.symbols

        fingerprint = np.zeros((len(positions), len(self._invariants)))

        nlist = ase.neighborlist.NeighborList(
            cutoffs=tuple(map(lambda sym: 0.5 * self._cutoffs[sym], symbols)), self_interaction=False, bothways=True
        )
        nlist.update(system)

        # Calculate the fingerprints for each atom centre one at a time
        for i, (pos, symbol) in enumerate(zip(positions, symbols)):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol], self._cutoff_fns[symbol])

            # Add ourselves
            env.add_gaussian(pos, sigma=self._sigmas[symbol], weight=self._weights[symbol])

            # Visit all neighbours
            indices, offsets = nlist.get_neighbors(i)
            for j, offset in zip(indices, offsets):
                neighbour_pos = positions[j] + np.dot(offset, system.get_cell())
                neighbour_symbol = symbols[j]
                env.add_gaussian(
                    neighbour_pos, sigma=self._sigmas[neighbour_symbol], weight=self._weights[neighbour_symbol]
                )

            fingerprint[i, :] = env.calc_moment_invariants(self._invariants, normalise=self._normalise)

        return fingerprint

    def create_single_old(self, system: ase.Atoms, index):
        # pylint: disable=too-many-locals

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
        fingerprint = np.zeros((n_types, len(self._invariants)))

        # Create the environments
        environs = []

        for _ in range(n_types):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol], self._cutoff_fns[symbol])
            # Add ourselves
            env.add_gaussian(pos, sigma=self._sigmas[symbol], weight=self._weights[symbol])
            environs.append(env)

        # Create the neighbour list
        nlist = ase.neighborlist.NeighborList(
            cutoffs=tuple(map(lambda sym: 0.5 * self._cutoffs[sym], symbols)), self_interaction=False, bothways=True
        )
        nlist.update(system)

        # Go through all the neighbours and add Gaussians on each
        indices, offsets = nlist.get_neighbors(index)
        for i, offset in zip(indices, offsets):
            neighbour_pos = positions[i] + np.dot(offset, system.get_cell())
            neighbour_symbol = symbols[i]

            for type_idx, types in enumerate(neighbour_types):
                if neighbour_symbol in types:
                    environs[type_idx].add_gaussian(
                        neighbour_pos, sigma=self._sigmas[neighbour_symbol], weight=self._weights[neighbour_symbol]
                    )

        # Finally compute all the fingerprints
        for type_idx, env in enumerate(environs):
            fingerprint[type_idx] = env.calc_moment_invariants(self._invariants, normalise=self._normalise)

        if len(fingerprint) == 1:
            return fingerprint[0]

        return fingerprint

    def create_single(self, system: ase.Atoms, index):
        # pylint: disable=too-many-locals

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
        fingerprint = np.zeros((n_types, len(self._invariants)))

        # Create the environments
        environs = []

        for _ in range(n_types):
            env = envs.SmoothGaussianEnvironment(pos, self._cutoffs[symbol], self._cutoff_fns[symbol])
            # Add ourselves
            env.add_gaussian(pos, sigma=self._sigmas[symbol], weight=self._weights[symbol])
            environs.append(env)

        if any(system.get_pbc()):
            dist_calculator = distances.UnitCellDistanceCalculator(system.cell.array)
        else:
            dist_calculator = distances.AperiodicDistanceCalculator()

        for i, neighbour_pos in enumerate(positions):
            neighbour_symbol = symbols[i]
            vecs = dist_calculator.get_vecs_between(pos, neighbour_pos, cutoff=cutoff, self_interation=i != index)
            for type_idx, types in enumerate(neighbour_types):
                if neighbour_symbol in types:
                    for vec in vecs:
                        environs[type_idx].add_gaussian(
                            pos + vec, sigma=self._sigmas[neighbour_symbol], weight=self._weights[neighbour_symbol]
                        )

        # Finally compute all the fingerprints
        for type_idx, env in enumerate(environs):
            fingerprint[type_idx] = env.calc_moment_invariants(self._invariants, normalise=self._normalise)

        if len(fingerprint) == 1:
            return fingerprint[0]

        return fingerprint


def extract_environments(
    system: ase.Atoms,
    atom_centered=True,
    cutoff=5.,
    yield_indices=False,
    include_central_atom=True,
):
    """Given an ase.Atoms this will extract atomic environments and yield them as new Atoms objects

    The central atom will always be at position 0 and the rest (if any) will follow.

    :param system: the atoms object to extract environments from
    :param atom_centered: if True will centre the new environments on the central (position 0) atom
    :param cutoff: a radial cutoff for defining each environment
    :param yield_indices: if True will yield a tuple (idx, Atoms) where idx is the global index of the central atom
    """
    # pylint: disable=too-many-locals

    positions = system.positions
    symbols = system.symbols

    # Create a distance calculator
    if any(system.get_pbc()):
        dist_calculator = distances.UnitCellDistanceCalculator(system.cell.array)
    else:
        dist_calculator = distances.AperiodicDistanceCalculator()

    # Go over each atom and create the environment
    for i, (central_pos, central_symbol) in enumerate(zip(positions, symbols)):
        env_positions = []
        env_symbols = []
        orig_indices = []
        if include_central_atom:
            env_positions.append(central_pos if not atom_centered else np.zeros(3))
            env_symbols.append(central_symbol)
            orig_indices.append(i)

        # Visit all neighbours
        for j, neighbour_pos in enumerate(positions):
            neighbour_symbol = symbols[j]
            vecs = dist_calculator.get_vecs_between(central_pos, neighbour_pos, cutoff=cutoff, self_interation=i != j)

            # Get all vectors to that neighbour (could be more than one if periodic)
            for vec in vecs:
                # Add the position
                env_positions.append(vec if atom_centered else vec + central_pos)
                # ...and the symbol
                env_symbols.append(neighbour_symbol)
                # Keep track of the index in the original structure
                orig_indices.append(j)

        atoms = ase.Atoms(positions=env_positions, symbols=env_symbols)
        atoms.set_array('orig_indices', np.array(orig_indices, dtype=int))

        # Finally yield the environment
        if yield_indices:
            yield i, atoms
        else:
            yield atoms


def ase2milad(atoms: ase.Atoms) -> atomic.AtomsCollection:
    return atomic.AtomsCollection(len(atoms), positions=atoms.positions, numbers=atoms.numbers)


def milad2ase(atoms: atomic.AtomsCollection) -> ase.Atoms:
    return ase.Atoms(positions=atoms.positions, numbers=atoms.numbers)


def prepare_molecule(*molecules: ase.Atoms, offset=0.) -> float:
    """This will bring the centroid of each molecule to be coincident with the origin and return
    the maximum radius found from any of the molecules"""
    max_radius_sq = 0.
    offset_vector = offset * np.array([1., 0., 0.])
    for molecule in molecules:
        centre, _radius = miniball.get_bounding_ball(molecule.positions)  # pylint: disable=unpacking-non-sequence
        centroid = centre + offset_vector
        # Set the new positions
        molecule.set_positions(molecule.positions - centroid)
        new_positions = molecule.positions
        # Get the maximum radius squared
        max_dist_sq = max(np.dot(pos, pos) for pos in new_positions)
        max_radius_sq = max(max_radius_sq, max_dist_sq)

    return max_radius_sq**0.5
