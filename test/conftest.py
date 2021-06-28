# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest

import milad
from milad import invariants, generate

READ_MAX = 128  # The maximum number of invariants to read from the file

# pylint: disable=redefined-outer-name, invalid-name


@pytest.fixture(scope='session')
def moment_invariants():
    """Get geometric moment invariants"""
    invs = milad.invariants.read(filename=invariants.GEOMETRIC_INVARIANTS, read_max=READ_MAX)
    yield invs


@pytest.fixture(scope='session')
def complex_invariants():
    """Get complex moment invariants"""
    invs = milad.invariants.read(filename=invariants.COMPLEX_INVARIANTS)
    yield invs


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(1234)
    np.random.seed(1234)


@pytest.fixture()
def save_figures():
    """Return True if you want to save figures from tests"""
    return False


@pytest.fixture()
def chiral_tetrahedra():
    """Create chiral structures that cannot be distinguished by bispectrum from
    https://link.aps.org/doi/10.1103/PhysRevLett.125.166001
    """
    return generate.chiral_tetrahedra()


@pytest.fixture()
def training_data():
    return create_fake_training_data()


@pytest.fixture()
def descriptor(complex_invariants):
    return create_descriptor(complex_invariants)


def create_descriptor(invs, _species=('Si',)):
    cutoff = 1.

    specie_numbers = (14,)
    return milad.descriptor(
        species=dict(map=dict(numbers=specie_numbers, range=(0.5, 1.5))),
        features=dict(type=milad.functions.WeightedDelta, map_species_to='WEIGHT'),
        moments_calculator=milad.zernike.ZernikeMomentCalculator(invs.max_order),
        invs=invs,
        cutoff=cutoff,
        apply_cutoff=True,
        smooth_cutoff=True
    )


def create_fake_training_data(num_atoms=10, max_atoms=10, species=('Si',)):
    import ase
    from ase.calculators.singlepoint import SinglePointCalculator

    training_data = []
    for _ in range(num_atoms):
        natoms = random.randint(1, max_atoms)

        atoms = ase.Atoms(positions=generate.random_points_in_sphere(natoms), symbols=random.choices(species, k=natoms))
        calc = SinglePointCalculator(atoms, energy=-100 * random.random(), forces=np.random.rand(natoms, 3))
        atoms.calc = calc

        training_data.append(atoms)

    return training_data
