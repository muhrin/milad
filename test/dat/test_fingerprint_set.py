# -*- coding: utf-8 -*-
from milad import dat
from milad import descriptors


def test_fingerprint_set(descriptor: descriptors.Descriptor, training_data):
    fingerprint_set = dat.create_fingerprint_set(descriptor, training_data, get_derivatives=True)

    assert descriptor.fingerprint_len == fingerprint_set.fingerprint_len
    assert fingerprint_set.has_all_forces()
    assert fingerprint_set.has_all_derivatives()

    total_envs = sum([len(atoms) for atoms in training_data])
    assert total_envs == fingerprint_set.total_environments

    # Test fingerprints appear as we would expect
    assert len(training_data) == len(fingerprint_set.fingerprints)
    for atoms, fps in zip(training_data, fingerprint_set.fingerprints):
        # One FP per atom
        assert len(fps) == len(atoms)

    assert all([len(atoms) == size for atoms, size in zip(training_data, fingerprint_set.sizes)])

    # Potential energies
    for atoms, energy in zip(training_data, fingerprint_set.get_potential_energies()):
        assert atoms.get_potential_energy() / len(atoms) == energy
