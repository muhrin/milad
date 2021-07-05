# -*- coding: utf-8 -*-
import numpy as np
import sympy

import milad
from milad import atomic
from milad import functions
from milad import invariants

# pylint: disable=invalid-name


def test_descriptor_derivatives():
    invs = invariants.read(invariants.COMPLEX_INVARIANTS, max_order=3)

    descriptor = milad.descriptor(
        features=dict(type=functions.WeightedDelta, map_species_to=None), cutoff=5., apply_cutoff=False, invs=invs
    )

    # Create a structure with just one atom, all symbolic
    r = sympy.IndexedBase('r', real=True)  # The positions
    s = sympy.IndexedBase('s', real=True)  # The species

    structure = atomic.AtomsCollection(
        1, positions=np.array([r[1], r[2], r[3]]), numbers=np.array([s[1]]), dtype=type(r[1])
    )
    phi, jac = descriptor(structure, jacobian=True)

    for i, entry in enumerate(phi):
        for j in range(3):
            diff = sympy.diff(entry, r[j + 1]).expand()
            from_jac = jac[i, j].expand()
            difference = diff - from_jac
            if not diff == from_jac:
                # If they differ, check that it's by a meaningful amount
                coeffs = np.array(tuple(difference.as_coefficients_dict().values()))
                np.testing.assert_array_almost_equal(coeffs, 0., decimal=10)


def test_descriptor_smoothness(descriptor):
    last_max = 0.
    for x in np.linspace(1.01, 0.9, 12):
        atoms = milad.AtomsCollection(1, positions=[x, 0., 0.], numbers=1)
        fingerprint = descriptor(atoms)

        fingerprint_max = np.abs(fingerprint).max()

        if x >= 1.:
            assert fingerprint_max == 0.

        assert fingerprint_max >= last_max
        print(x, fingerprint_max)

        last_max = fingerprint_max
