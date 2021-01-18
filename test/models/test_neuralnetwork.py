# -*- coding: utf-8 -*-
import ase
import numpy as np

from milad import models
from milad import generate
from milad import fingerprinting


def test_create_fingerprint_set():
    # Settings
    np.random.seed(5)
    num_atoms = 6
    cutoff = 2.5
    descriptor = fingerprinting.descriptor(cutoff=cutoff)

    positions = generate.random_points_in_sphere(num_atoms, cutoff)
    atoms = ase.Atoms(positions=positions, numbers=(1.,) * num_atoms)

    fingerprint_set = models.create_fingerprint_set(descriptor, [atoms], get_derivatives=True)
    assert len(fingerprint_set) == 1
