# -*- coding: utf-8 -*-
# import numpy as np
#
# from milad import reconstruct, fingerprinting, atomic
#
#  Skipping this for now
# def test_find_from_moments(complex_invariants):
#     num_atoms = 10
#
#     descriptor = fingerprinting.descriptor(invs=complex_invariants, cutoff=1.0)
#
#     atoms = atomic.random_atom_collection_in_sphere(num_atoms)
#     mask = atoms.get_mask()
#     mask.numbers[:] = 1.
#     moments = descriptor.get_moments(atoms)
#
#     res = reconstruct.find_atoms_from_moments(
#         descriptor,
#         moments,
#         num_atoms,
#         mask=mask,
#         verbose=True,
#     )
#
#     assert np.allclose(atoms.positions, res.value.positions)
