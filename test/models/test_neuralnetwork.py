# -*- coding: utf-8 -*-
import ase
import torch
from ase.calculators import singlepoint
import numpy as np
import pytest

from milad import models
from milad.models import neuralnetwork
from milad import generate
from milad import fingerprinting
from milad import utils


@pytest.fixture
def fake_data() -> utils.FingerprintSet:
    # Settings
    np.random.seed(5)
    num_atoms = 6
    cutoff = 2.5
    descriptor = fingerprinting.descriptor(cutoff=cutoff)

    positions = generate.random_points_in_sphere(num_atoms, cutoff)
    atoms = ase.Atoms(positions=positions, numbers=(1.,) * num_atoms)
    atoms.set_array('force', np.zeros((6, 3)))
    calc = singlepoint.SinglePointCalculator(atoms, energy=-10)
    atoms.set_calculator(calc)

    return models.create_fingerprint_set(descriptor, [atoms], get_derivatives=True)


def test_create_fingerprint_set(fake_data):
    assert len(fake_data) == 1


def test_neural_network_basics(fake_data: utils.FingerprintSet):
    nn = models.NeuralNetwork()
    nn._create_network(fake_data.fingerprint_len)
    fitting_data = neuralnetwork.FittingData.from_fingerprint_set(fake_data, requires_grad=True)
    res = nn.make_prediction(fitting_data, get_forces=True)

    # Check that the answer is the same with standard or vectorised Jacobian calculation
    nn.vectorise_jacobian = True
    res_vec = nn.make_prediction(fitting_data, get_forces=True)
    assert torch.allclose(res.forces, res_vec.forces)
    assert torch.allclose(res.local_energies, res_vec.local_energies)
