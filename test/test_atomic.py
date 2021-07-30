# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest
from scipy import spatial
import sympy

from milad import atomic
from milad import optimisers
from milad import testing


def test_map_numbers():
    """Test that the mapping atomic numbers function works correctly"""
    num_atoms = 6
    system = atomic.AtomsCollection(
        num_atoms, positions=np.random.rand(num_atoms, 3), numbers=np.array([1, 2, 5, 2, 3, 8])
    )
    numbers = set(system.numbers)

    # Map the five species into the range 2 -> 3
    mapper = atomic.MapNumbers(numbers, map_to=(2., 3.))
    mapped = mapper(system)
    assert mapped.numbers[0] == pytest.approx(2.1)
    assert mapped.numbers[1] == pytest.approx(2.3)
    assert mapped.numbers[2] == pytest.approx(2.7)
    assert mapped.numbers[3] == pytest.approx(2.3)
    assert mapped.numbers[4] == pytest.approx(2.5)
    assert mapped.numbers[5] == pytest.approx(2.9)

    # Now test inversion
    inverted = mapper.inverse(mapped)
    diffs = inverted.numbers - system.numbers
    assert diffs.max() == pytest.approx(0.)

    testing.test_function(mapper, system.copy())

    # Now check that we can also map all species down to a single number (i.e. discard species info)
    mapper = atomic.MapNumbers(numbers, map_to=3.4)
    mapped = mapper(system)
    assert np.all(mapped.numbers == 3.4)


def test_separation_force():
    """Test that the separation force works correctly"""
    separation_force = atomic.SeparationForce()
    separation = 0.5
    atoms = atomic.AtomsCollection(2, positions=[[-0.5 * separation, 0, 0], [0.5 * separation, 0, 0]], numbers=1.)

    energy, jacobian = separation_force(atoms, jacobian=True)
    assert energy == separation_force.energy(separation)
    assert energy > 0., 'The energy should always be positive if the separation less than the cutoff'

    force = separation_force.force(separation)
    assert force > 0., 'The force should act to separate the two atoms'

    assert np.all(jacobian[0, atoms.linear_pos_idx(0)] == np.array([force, 0., 0.]))
    assert np.all(jacobian[0, atoms.linear_pos_idx(1)] == np.array([-force, 0., 0.]))

    # Check that the cutoff works
    atoms.positions[1, 0] = 1.
    energy, jacobian = separation_force(atoms, jacobian=True)
    assert energy == 0.0
    assert np.all(jacobian == 0.)


def test_separation_force_symbolic():
    # pylint: disable=invalid-name
    force = atomic.SeparationForce(cutoff=1.)
    # Manually set the cutoff to None because otherwise the symbolic derivative check can't decide if
    # it is or isn't within the cutoff
    force._cutoff = None  # pylint: disable=protected-access

    r = sympy.Symbol('r', real=True)
    energy = force.energy(r)
    derivative = sympy.diff(energy, r)
    assert force.force(r) == -derivative


def test_separation_force_optimiser():
    """Test using the separation force to actually separate some atoms using gradient descent"""
    cutoff = 1.
    force = atomic.SeparationForce(epsilon=0.1, sigma=0.5, cutoff=cutoff)
    atoms = atomic.random_atom_collection_in_sphere(10, radius=0.3)

    res = optimisers.LeastSquaresOptimiser().optimise(
        force,
        initial=atoms,
        # bounds=(-1., 1.),
    )

    assert res.success
    distances = spatial.distance.pdist(res.value.positions)
    assert np.all(distances > cutoff)


def test_scale_positions():
    atoms = atomic.random_atom_collection_in_sphere(10, radius=5., centre=True)
    testing.test_function(atomic.ScalePositions(0.5), atoms, check_jacobian=True)


def test_feature_mapper():
    atoms = atomic.random_atom_collection_in_sphere(10, numbers=tuple(random.randint(0, 100) for _ in range(10)))
    testing.test_function(atomic.FeatureMapper(map_species_to='WEIGHT'), atoms, check_jacobian=True)
