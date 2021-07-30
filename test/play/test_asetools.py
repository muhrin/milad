# -*- coding: utf-8 -*-
import math

import ase.build
import ase.io
import ase.neighborlist
import matplotlib.pyplot as plt
import numpy as np

import milad
from milad.play import asetools

# pylint: disable=invalid-name


def test_multiple_species(moment_invariants, request, save_figures):
    molecule = ase.build.molecule('CH3CH2OH')
    species = ['C', 'H', 'O']
    symbols = molecule.symbols
    num_atoms = len(symbols)

    fp = asetools.MiladFingerprint(species, moment_invariants, sigmas={'C': 1., 'H': 0.5, 'O': 0.7})

    fingerprints = fp.create(molecule)

    labels = []
    for i in range(num_atoms):
        labels.append('{}$_{}$'.format(i, symbols[i]))

    assert (fingerprints[3] - fp.create_single(molecule, 3)).max() == 0

    fig, axes = plt.subplots()
    axes.set_yscale('log')
    milad.plot.plot_multiple_invariants(fingerprints, axes, labels=labels)
    axes.set_title('C$_2$H$_6$O')
    fig.legend()
    if save_figures:
        fig.savefig('{}.pdf'.format(request.node.name))


def test_multiple_species_split(moment_invariants, request, save_figures):
    molecule = ase.build.molecule('CH3CH2OH')
    species = ['C', 'H', 'O']

    fp = asetools.MiladFingerprint(
        species, moment_invariants, sigmas={
            'C': 0.4,
            'H': 0.3,
            'O': 0.5
        }, split_specie_pairs=True, cutoffs=100.
    )

    fingerprint = fp.create_single(molecule, 2)

    fig, axes = plt.subplots()
    axes.set_yscale('log')
    axes.set_title(str(molecule))
    milad.plot.plot_multiple_invariants(fingerprint, axes, labels=species)
    fig.legend()
    if save_figures:
        fig.savefig('{}.pdf'.format(request.node.name))


def test_generate_environments_molecule():
    molecule = ase.build.molecule('CH3CH2OH')

    envs = list(asetools.extract_environments(molecule, cutoff=4.))
    assert len(envs) == len(molecule)


def test_generate_environments_solid():
    # pylint: disable=too-many-nested-blocks, too-many-locals
    lattice_param = 2.5
    cutoff = 4.

    diamond = ase.build.bulk('C', 'hcp', a=lattice_param)
    envs = list(asetools.extract_environments(diamond, cutoff=cutoff))

    lattice_max = int(math.ceil(lattice_param / cutoff) + 1)

    cutoff_sq = cutoff * cutoff
    # lattice params
    a, b, c = diamond.cell[0], diamond.cell[1], diamond.cell[2]

    found_positions = []
    for i, central in enumerate(diamond.positions):
        env_positions = []
        for _j, other in enumerate(diamond.positions):
            r_ij = other - central

            for l in range(-lattice_max, lattice_max + 1):
                for m in range(-lattice_max, lattice_max + 1):
                    for n in range(-lattice_max, lattice_max + 1):
                        # if l == m == n == 0:
                        #     continue

                        dr = r_ij + l * a + m * b + n * c
                        if np.dot(dr, dr) < cutoff_sq:
                            env_positions.append(dr)

        known_positions = envs[i].positions
        assert len(env_positions) == len(known_positions)
        positions = np.array(env_positions)
        found_positions.append(positions)

        # Check the list of distances matches at least
        assert sorted(np.sum(positions * positions, axis=1)) == \
               sorted(np.sum(known_positions * known_positions, axis=1))

    assert len(envs) == len(found_positions)


def test_extract_environments():
    num_atoms = 5
    cutoff = 0.7
    system = ase.Atoms(
        positions=[[0., 0., 0.]] * num_atoms, numbers=[6] * num_atoms, cell=[1, 1, 1, 47, 97, 121], pbc=True
    )
    system.rattle(stdev=156)
    system.wrap()

    nlist = ase.neighborlist.NeighborList([cutoff / 2.] * num_atoms, skin=0., self_interaction=False, bothways=True)
    nlist.update(system)

    for i, env in tuple(
        asetools.extract_environments(system, cutoff=cutoff, yield_indices=True, include_central_atom=False)
    ):
        indices, offsets = nlist.get_neighbors(i)
        pos = system.positions[i]

        Rs = np.array([
            system.positions[j] + np.dot(offset, system.cell) - pos for (j, offset) in zip(indices, offsets)
        ])

        indices = sorted(indices)
        env_indices = sorted(env.get_array('orig_indices'))
        assert env_indices == indices

        dots = np.array(sorted(np.sum(Rs * Rs, axis=1)))
        env_dots = np.array(sorted(np.sum(env.positions * env.positions, axis=1)))

        assert np.allclose(dots, env_dots)
