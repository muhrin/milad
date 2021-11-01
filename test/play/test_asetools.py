# -*- coding: utf-8 -*-
import math

import ase.build
import ase.io
import ase.neighborlist
import numpy as np

from milad.play import asetools

# pylint: disable=invalid-name


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
