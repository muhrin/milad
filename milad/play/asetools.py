# -*- coding: utf-8 -*-
from typing import Union, Any

import ase.neighborlist
import miniball
import numpy as np
import numpy.linalg

from milad import atomic
from . import distances


def _create_params_dict(species: list, value: Union[Any, dict]) -> dict:
    if isinstance(value, dict):
        return value

    return {specie: value for specie in species}


def extract_environments(
    system: ase.Atoms,
    atom_centered=True,
    cutoff=5.0,
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
            vecs = dist_calculator.get_vecs_between(
                central_pos, neighbour_pos, cutoff=cutoff, self_interation=i != j
            )

            # Get all vectors to that neighbour (could be more than one if periodic)
            for vec in vecs:
                # Add the position
                env_positions.append(vec if atom_centered else vec + central_pos)
                # ...and the symbol
                env_symbols.append(neighbour_symbol)
                # Keep track of the index in the original structure
                orig_indices.append(j)

        atoms = ase.Atoms(positions=env_positions, symbols=env_symbols)
        atoms.set_array("orig_indices", np.array(orig_indices, dtype=int))

        # Finally yield the environment
        if yield_indices:
            yield i, atoms
        else:
            yield atoms


def ase2milad(atoms: ase.Atoms) -> atomic.AtomsCollection:
    return atomic.AtomsCollection(
        len(atoms), positions=atoms.positions, numbers=atoms.numbers
    )


def milad2ase(atoms: atomic.AtomsCollection) -> ase.Atoms:
    return ase.Atoms(positions=atoms.positions, numbers=atoms.numbers)


def centre_molecule(molecule: ase.Atoms) -> float:
    molecule.positions[:] = molecule.positions / len(molecule)
    norms = np.linalg.norm(molecule.positions, axis=0)
    return norms.max()


def prepare_molecule(*molecules: ase.Atoms) -> float:
    """This will bring the centroid of each molecule to be coincident with the origin and return
    the maximum radius found from any of the molecules"""
    max_radius_sq = 0.0
    for molecule in molecules:
        try:
            centre, _radius = miniball.get_bounding_ball(
                molecule.positions
            )  # pylint: disable=unpacking-non-sequence
        except numpy.linalg.LinAlgError:
            max_dist_sq = centre_molecule(molecule)
        else:
            centroid = centre
            # Set the new positions
            molecule.set_positions(molecule.positions - centroid)
            new_positions = molecule.positions
            # Get the maximum radius squared
            max_dist_sq = max(np.dot(pos, pos) for pos in new_positions)

        max_radius_sq = max(max_radius_sq, max_dist_sq)

    return max_radius_sq**0.5
