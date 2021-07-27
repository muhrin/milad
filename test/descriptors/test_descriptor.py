# -*- coding: utf-8 -*-
import numpy as np

import milad
from milad import atomic
from milad import functions
from milad import invariants
from milad import testing

# pylint: disable=invalid-name


def test_descriptor_derivatives():
    invs = invariants.read(invariants.COMPLEX_INVARIANTS, max_order=3)

    descriptor = milad.descriptor(
        features=dict(type=functions.WeightedDelta, map_species_to=None), cutoff=5., apply_cutoff=False, invs=invs
    )

    structure = atomic.random_atom_collection_in_sphere(1, 5.)
    testing.test_function(descriptor, structure)


def test_descriptor_smoothness(descriptor):
    last_max = 0.
    for x in np.linspace(1.01, 0.9, 12):
        atoms = milad.AtomsCollection(1, positions=[x, 0., 0.], numbers=1)
        fingerprint = descriptor(atoms)

        fingerprint_max = np.abs(fingerprint).max()

        if x >= 1.:
            assert fingerprint_max == 0.

        assert fingerprint_max >= last_max

        last_max = fingerprint_max
