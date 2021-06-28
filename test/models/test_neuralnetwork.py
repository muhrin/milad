# -*- coding: utf-8 -*-
import torch
import numpy as np

from milad import dat
from milad.models import neuralnetwork

# pylint: disable=invalid-name, redefined-outer-name


def test_fitting_data(descriptor, training_data):
    fingerprint_set = dat.create_fingerprint_set(descriptor, training_data, get_derivatives=True)
    fitting_data = neuralnetwork.FittingData.from_fingerprint_set(fingerprint_set, requires_grad=True)

    num_systems = len(training_data)
    total_environments = sum([len(atoms) for atoms in training_data])

    assert len(fitting_data) == num_systems
    assert fitting_data.fingerprint_len == fingerprint_set.fingerprint_len
    assert len(fitting_data.fingerprints) == total_environments

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
    fitting_data = neuralnetwork.FittingData.from_fingerprint_set(fingerprint_set, requires_grad=True)

    nn.init(fitting_data)

    predictions = nn.make_prediction(fitting_data)
    assert torch.all(predictions.indices == fitting_data.index)
    assert torch.all(predictions.sizes == fitting_data.num_atoms)


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
