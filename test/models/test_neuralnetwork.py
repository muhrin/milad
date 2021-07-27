# -*- coding: utf-8 -*-
import ase
from ase.calculators import singlepoint
import pytest
import torch
import numpy as np

from milad import dat
from milad.models import neuralnetwork

torch.set_default_dtype(torch.float64)

# pylint: disable=invalid-name

SIGMA = 0.4
EPSILON = 4.


def lennard_jones(r):
    return EPSILON * ((SIGMA / r)**12 - (SIGMA / r)**6)


def lennard_jones_force(r):
    """Get magnitude of LJ force"""
    return -EPSILON * ((-12 * SIGMA**12 / r**13) + (6 * SIGMA**6 / r**7))


@pytest.fixture
def lj_training_data():
    training_size = 50
    r = np.linspace(0.3, 1., num=training_size)

    # Create training energies
    training_energies = np.array(list(map(lennard_jones, r)))
    # Create forces
    drs = np.outer(r, np.array([1.0, 0, 0]))  # Create a bunch of vectors that point along positive-x
    training_forces = np.array([force * dr for force, dr in zip(map(lennard_jones_force, r), drs)])

    training_data = []
    for dr, energy, force in zip(drs, training_energies, training_forces):
        atoms = ase.Atoms(positions=[[0., 0., 0.], dr], numbers=[6, 6])
        calc = singlepoint.SinglePointCalculator(atoms, energy=energy, forces=[-force, force])
        atoms.calc = calc

        training_data.append(atoms)

    return training_data


# pylint: disable=invalid-name, redefined-outer-name


def test_fitting_data(descriptor, training_data):
    fingerprint_set = dat.create_fingerprint_set(descriptor, training_data, get_derivatives=True)
    fitting_data = neuralnetwork.FittingData.from_fingerprint_set(fingerprint_set, requires_grad=True)

    num_systems = len(training_data)
    total_environments = sum([len(atoms) for atoms in training_data])

    assert len(fitting_data) == num_systems
    assert fitting_data.fingerprint_len == fingerprint_set.fingerprint_len
    assert len(fitting_data.fingerprints) == total_environments

    # Check that iterating system-wise works correctly
    idx = 0
    for system_idx, system in enumerate(fitting_data.iter_systemwise()):
        natoms = fitting_data.num_atoms[system_idx]
        assert fitting_data.total_energies[system_idx] == system.total_energy
        assert torch.all(fitting_data.derivatives[system_idx] == system.derivatives)

        assert torch.all(fitting_data.fingerprints[idx:idx + natoms] == system.fingerprints)
        assert torch.all(fitting_data.forces[idx:idx + natoms] == system.forces)
        assert torch.all(fitting_data.index[idx:idx + natoms] == system.index)

        idx += natoms

    total_atoms = 0
    for idx, atoms in enumerate(training_data):
        natoms = len(atoms)

        assert np.isclose(fitting_data.num_atoms[idx].item(), natoms)
        assert np.isclose(fitting_data.total_energies[idx].item(), atoms.get_potential_energy())
        assert np.isclose(fitting_data.get_normalised_energies()[idx].item(), atoms.get_potential_energy() / natoms)
        assert np.all(fitting_data.index[total_atoms:total_atoms + natoms].detach().numpy() == idx)

        total_atoms += natoms

    assert fitting_data.batch_split(len(training_data))[0] is fitting_data


def test_neural_network_basics(descriptor, training_data):
    nn = neuralnetwork.NeuralNetwork()
    fingerprint_set = dat.create_fingerprint_set(descriptor, training_data, get_derivatives=True)
    fitting_data = neuralnetwork.FittingData.from_fingerprint_set(fingerprint_set, requires_grad=True, device=nn.device)

    nn.init(fitting_data)

    predictions = nn.make_prediction(fitting_data, get_forces=True)
    assert torch.all(predictions.indices == fitting_data.index)
    assert torch.all(predictions.sizes == fitting_data.num_atoms)
    assert len(predictions.forces) == len(fitting_data.fingerprints), 'Expect one force per environment'

    nn.fit(fitting_data, max_epochs=1)


def test_predictions(descriptor, training_data):
    fingerprint_set = dat.create_fingerprint_set(descriptor, training_data, get_derivatives=True)
    fitting_data = neuralnetwork.FittingData.from_fingerprint_set(fingerprint_set, requires_grad=True)

    # Artificially create per-atom energies (just split the total energy evenly
    env_energies = torch.zeros((fitting_data.index.shape[0], 1), dtype=fitting_data.total_energies.dtype)
    total_atoms = 0
    for idx, energy in enumerate(fitting_data.get_normalised_energies()):
        natoms = fitting_data.num_atoms[idx]
        env_energies[total_atoms:total_atoms + natoms, 0] = energy
        total_atoms += natoms

    pred = neuralnetwork.Predictions(fitting_data.num_atoms, fitting_data.index, env_energies)
    predicted_energies = pred.get_normalised_energies()
    known_energies = fitting_data.get_normalised_energies()
    assert torch.allclose(predicted_energies, known_energies)

    loss_fn = neuralnetwork.LossFunction()
    loss = loss_fn.get_loss(pred, fitting_data)
    assert torch.isclose(loss.energy, torch.tensor(0.))
    assert torch.isclose(loss.total, torch.tensor(0.))


def test_nn_lennard_jones(descriptor, lj_training_data):
    """Check if we can successfully fit to the LJ potential"""
    nn = neuralnetwork.NeuralNetwork(hiddenlayers=(10, 10))

    fingerprint_set = dat.create_fingerprint_set(descriptor, lj_training_data, get_derivatives=True)
    _training, _loss_fn = nn.fit(fingerprint_set, max_epochs=50, batchsize=1000, learning_rate=1e-3)
