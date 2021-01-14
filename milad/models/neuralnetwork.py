# -*- coding: utf-8 -*-
import math
from typing import Union, Callable, Sequence, Tuple

import ase
import torch
import torch.nn.functional

from milad.play import asetools
from milad import fingerprinting
from milad import utils

__all__ = 'NeuralNetwork', 'create_fingerprint_set'


def create_fingerprint_set(
    descriptor: fingerprinting.MomentInvariantsDescriptors, systems: Sequence[ase.Atoms]
) -> utils.FingerprintSet:
    fingerprints = utils.FingerprintSet(descriptor.fingerprint_len)
    for system in systems:
        fps = []
        for env in asetools.extract_environments(system, cutoff=descriptor.cutoff):
            fps.append(descriptor(asetools.ase2milad(env)))
        fingerprints.add_system(system, fps)

    return fingerprints


class FittingData:
    """Data used during a fitting procedure"""

    @classmethod
    def from_fingerprint_set(cls, fingerprint_set: utils.FingerprintSet, device=None, dtype=None, requires_grad=False):
        tensor_kwargs = dict(dtype=dtype, device=device, requires_grad=requires_grad)
        fingerprints = tuple(torch.tensor(fps, **tensor_kwargs) for fps in fingerprint_set.fingerprints)
        energies = torch.tensor(fingerprint_set.get_potential_energies(normalise=False), **tensor_kwargs)
        return FittingData(fingerprints, energies)

    def __init__(self, fingerprints: Tuple[torch.Tensor], energies: torch.Tensor):
        self._fingerprints = fingerprints
        self._energies = energies
        self._num_atoms = torch.tensor(tuple(len(fps) for fps in fingerprints), device=fingerprints[0].device)

    def __getitem__(self, item) -> 'FittingData':
        if isinstance(item, slice):
            return FittingData(self._fingerprints[item], self._energies[item])

        fingerprints = (self._fingerprints[item],)
        energies = self._energies[item].reshape(1, 1)
        return FittingData(fingerprints, energies)

    @property
    def fingerprints(self) -> Tuple[torch.Tensor]:
        return self._fingerprints

    @property
    def energies(self) -> torch.Tensor:
        """Get the known energies"""
        return self._energies

    @property
    def num_atoms(self) -> torch.Tensor:
        """Get the number of atoms in each structure"""
        return self._num_atoms

    def get_normalised_energies(self) -> torch.Tensor:
        return self._energies / self._num_atoms


class LossFunction:

    def get_loss(self, predicted: torch.Tensor, target: torch.Tensor, num_atoms: torch.Tensor):
        return torch.nn.functional.mse_loss(predicted / num_atoms, target / num_atoms)


class Range:
    """A range object representing a minimum and maximum value"""

    def __init__(self, initial=(0., 1.), device=None, dtype=None):
        initial = initial or (0., 1.)
        self._range = torch.tensor(initial, device=device, dtype=dtype)
        self._using_defaults = True

    def __str__(self):
        return '[{}, {}]'.format(self.min.item(), self.max.item())

    @property
    def range(self):
        return self._range

    @property
    def min(self):
        return self._range[0]

    @property
    def max(self):
        return self._range[1]

    @property
    def span(self):
        return self.max - self.min

    def expand(self, values: torch.Tensor):
        """Expand the range to encompass the passed values"""
        if self._using_defaults:
            self._range[0] = values.min()
            self._range[1] = values.max()
            self._using_defaults = False
        else:
            vals_min = values.min()
            vals_max = values.max()
            self._range[0] = self.min if self.min < vals_min else vals_min
            self._range[1] = self.max if self.max > vals_max else vals_max


class DataScaler(torch.nn.Module):
    """Scale data to be in the range in a particular range.  Handles scaling on both input and output sides
    so if you can set the range of input and outputs separately."""

    def __init__(self, in_range=(0., 1.), out_range=(0., 1.), device=None, dtype=None):
        super().__init__()
        # Default to range [-1, 1]
        self._in_range = Range(in_range, device=device, dtype=dtype)  # Input range
        self._out_range = Range(out_range, device=device, dtype=dtype)  # Output range

    def __str__(self):
        return '{} -> {}'.format(self._in_range, self._out_range)

    @property
    def input(self) -> Range:
        return self._in_range

    @property
    def output(self) -> Range:
        return self._out_range

    def scale(self, x: torch.Tensor):  # pylint: disable=invalid-name
        return self.output.span * (x - self.input.min) / self.input.span + self.output.min

    def unscale(self, y: torch.Tensor):  # pylint: disable=invalid-name
        return (y - self.output.min) * self.input.span / self.output.span + self.input.min

    def forward(self, x: torch.Tensor):  # pylint: disable=invalid-name
        return self.scale(x)


class NeuralNetwork:
    """Neural network empirical potential model"""

    def __init__(
        self,
        hiddenlayers=(32, 16, 8),
        activations: Union[str, Callable] = 'ReLU',
        loss_function=None,
        device=None,
        dtype=torch.float64
    ):
        # Set up the hidden layers
        self._hidden = hiddenlayers

        # Set the activation function
        if isinstance(activations, str):
            self._activations = getattr(torch.nn, activations)()
        elif isinstance(activations, Callable):
            self._activations = activations
        else:
            raise TypeError('Expecting str or Callable, got {}'.format(activations.__class__.__name__))

        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device
        self._dtype = dtype

        self._fingerprint_scaler = DataScaler(out_range=(-1., 1.), device=self._device, dtype=self._dtype)
        self._energy_scaler = DataScaler(in_range=(-1., 1.), device=self._device, dtype=self._dtype)
        self.loss_function = loss_function or LossFunction()

        self._network = None

    def fit(
        self,
        training_set: Union[Sequence[ase.Atoms], utils.FingerprintSet],
        max_epochs=200,
        progress_callback: Callable = None,
        learning_rate=5e-4,
        batchsize=16,
    ):
        if self._network is None:
            self._create_network(training_set.fingerprint_len)

        training_data = self.create_fitting_data(training_set, requires_grad=True)

        # Scale the inputs and outputs to match the ranges
        # self._fingerprint_scaler.input.expand(training_data.fingerprints)
        norm_energies = training_data.get_normalised_energies()
        self._energy_scaler.output.expand(norm_energies)

        def stopping_function(epoch, _training, _loss):
            if epoch >= max_epochs:
                return True

            return False  # Don't stop

        optimiser = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        loss = self._train(training_data, stopping_function, optimiser, batchsize, progress_callback)
        return training_data, loss

    def create_fitting_data(self, fingerprint_set: utils.FingerprintSet, requires_grad=False):
        return FittingData.from_fingerprint_set(
            fingerprint_set, device=self._device, dtype=self._dtype, requires_grad=requires_grad
        )

    def _train(
        self,
        training: FittingData,
        should_stop: Callable,
        optimiser,
        batchsize,
        progress_callback: Callable = None,
    ):
        total_samples = len(training.fingerprints)
        batches = int(math.ceil(total_samples / batchsize))

        epoch = 0
        while True:
            for batch_num in range(batches):
                start_idx = batch_num * batchsize
                end_idx = min(start_idx + batchsize, total_samples)
                batch = training[start_idx:end_idx]

                predicted_energies = self._make_prediction(*batch.fingerprints)

                optimiser.zero_grad()
                loss = self.loss_function.get_loss(predicted_energies, batch.energies, batch.num_atoms)
                loss.backward(retain_graph=True)
                optimiser.step()

            predicted_energies = self._make_prediction(*training.fingerprints)
            loss = self.loss_function.get_loss(predicted_energies, training.energies, training.num_atoms)

            if progress_callback is not None:
                progress_callback(self, epoch, training, loss)

            if should_stop(epoch, training, loss):
                break

            # print("MADE STEP")
            epoch += 1

        return loss

    def loss(self, fitting_data: FittingData):
        predicted_energies = self._make_prediction(*fitting_data.fingerprints)
        return self.loss_function.get_loss(predicted_energies, fitting_data.energies, fitting_data.num_atoms)

    def _make_prediction(self, *fingerprints: torch.Tensor, normalise=False) -> torch.Tensor:
        # Join all fingerprints as this is much faster going through the network
        all_fingerprints = torch.cat(fingerprints, 0)
        local_energies = self._network(all_fingerprints)

        # Now calculate total energy for each system
        total_energies = []
        idx = 0
        for fps in fingerprints:
            natoms = fps.shape[0]
            summed = torch.sum(local_energies[idx:idx + natoms])
            summed *= 1. if not normalise else 1. / natoms
            total_energies.append(summed)
            idx += natoms

        stacked = torch.stack(total_energies)
        return stacked

    def _create_network(self, input_size):
        # Create the network
        # sequence = [self._fingerprint_scaler]
        sequence = []
        prev_size = input_size
        # Create the fully connected (hidden) layers
        for size in self._hidden:
            sequence.append(torch.nn.Linear(prev_size, size, bias=False))
            sequence.append(self._activations)
            prev_size = size
        # Add the output layer
        sequence.extend((torch.nn.Linear(prev_size, 1, bias=False), self._energy_scaler))

        self._network = torch.nn.Sequential(*sequence)
        self._network.to(self._device, self._dtype)

    def tensor(self, *args, requires_grad=False) -> torch.Tensor:
        return torch.tensor(*args, device=self._device, dtype=self._dtype, requires_grad=requires_grad)
