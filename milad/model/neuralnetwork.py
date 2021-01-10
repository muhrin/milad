# -*- coding: utf-8 -*-
from typing import Union, Callable, Sequence

import ase
import torch
import torch.nn.functional

from milad.play import asetools
from milad import fingerprinting
from milad import utils

__all__ = 'NeuralNetwork', 'create_fingerprint_set'


def create_fingerprint_set(
    descriptor: fingerprinting.Fingerprinter, systems: Sequence[ase.Atoms]
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

    def __init__(self, fingerprint_set: utils.FingerprintSet, device=None, dtype=None, requires_grad=False):
        self._fingerprint_set = fingerprint_set
        self._fingerprints = torch.tensor(
            fingerprint_set.fingerprints, device=device, dtype=dtype, requires_grad=requires_grad
        ).reshape(len(fingerprint_set), fingerprint_set.fingerprint_len)
        energies = fingerprint_set.get_potential_energies(normalise=True)
        self._energies = torch.tensor(energies, device=device, dtype=dtype,
                                      requires_grad=requires_grad).reshape((len(energies), 1))
        self.predicted_energies = None  # The last set of predicted energies

    @property
    def fingerprint_set(self) -> utils.FingerprintSet:
        return self._fingerprint_set

    @property
    def fingerprints(self) -> torch.Tensor:
        return self._fingerprints

    @property
    def energies(self) -> torch.Tensor:
        """Get the known energies"""
        return self._energies

    def make_prediction(self, network) -> torch.Tensor:
        local_energies = network(self.fingerprints)
        total_energies = self.fingerprint_set.systemwise_sum(local_energies, normalise=True)
        self.predicted_energies = torch.stack(total_energies)
        return self.predicted_energies

    def scatter(self, axes, **kwargs):
        axes.scatter(self.energies.cpu().detach().numpy(), self.predicted_energies.cpu().detach().numpy(), **kwargs)


class LossFunction:

    def get_loss(self, fitting_data: FittingData):
        return torch.nn.functional.mse_loss(fitting_data.predicted_energies, fitting_data.energies)


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
    """Scale data to be in the range [-1, 1]"""

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

    def scale(self, x: torch.Tensor):
        return self.output.span * (x - self.input.min) / self.input.span + self.output.min

    def unscale(self, y: torch.Tensor):
        return (y - self.output.min) * self.input.span / self.output.span + self.input.min

    def forward(self, x: torch.Tensor):
        return self.scale(x)


class NeuralNetwork:
    """Neural network empirical potential model"""

    def __init__(
        self,
        hiddenlayers=(16, 16, 16),
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
        descriptor: fingerprinting.Fingerprinter,
        training_set: Union[Sequence[ase.Atoms], utils.FingerprintSet],
        validation_set: Union[Sequence[ase.Atoms], utils.FingerprintSet] = None,
        max_epochs=200,
        progress_callback: Callable = None
    ):
        if not isinstance(training_set, utils.FingerprintSet):
            training_set = create_fingerprint_set(descriptor, training_set)
        if validation_set is not None and not isinstance(validation_set, utils.FingerprintSet):
            validation_set = create_fingerprint_set(descriptor, validation_set)

        if self._network is None:
            self._create_network(training_set.fingerprint_len)

        training_data = FittingData(training_set, device=self._device, dtype=self._dtype, requires_grad=True)
        validation_data = FittingData(
            validation_set, device=self._device, dtype=self._dtype
        ) if validation_set else None

        # Scale the inputs and outputs to match the ranges
        self._fingerprint_scaler.input.expand(training_data.fingerprints)
        self._energy_scaler.output.expand(training_data.energies)

        # if validation_set is not None:
        #     self._fingerprint_scaler.input.expand(validation_data.fingerprints)
        #     self._energy_scaler.output.expand(validation_data.energies)

        def stopping_function(step, _training, _loss, _validation, _validation_loss):
            if step >= max_epochs:
                return True

            return False  # Don't stop

        loss, validation_loss = self._train(training_data, stopping_function, validation_data, progress_callback)
        return training_data, loss, validation_data, validation_loss

    def _train(
        self,
        training: FittingData,
        should_stop: Callable,
        validation: FittingData = None,
        progress_callback: Callable = None
    ):
        optimiser = torch.optim.Adam(self._network.parameters(), lr=1e-4)
        validation_loss = None

        # dataset = tdata.TensorDataset(training.fingerprints, training.energies)
        # train_loader = tdata.DataLoader(dataset, batch_size=16, shuffle=True)

        step = 0
        while True:
            # print("Starting step")

            training.make_prediction(self._network)
            loss = self.loss_function.get_loss(training)
            # local_energies = self._network(training.fingerprints)
            # prediction = torch.stack(training.fingerprint_set.systemwise_sum(local_energies))
            # print(prediction.shape)
            # print(training.energies.shape)
            # loss = torch.nn.functional.mse_loss(prediction, training_energies)

            optimiser.zero_grad()
            loss.backward(retain_graph=True)
            optimiser.step()

            if validation:
                validation.make_prediction(self._network)
                validation_loss = self.loss_function.get_loss(validation)

            if progress_callback is not None:
                progress_callback(step, training, loss, validation, validation_loss)

            if should_stop(step, training, loss, validation, validation_loss):
                break

            # print("MADE STEP")
            step += 1

        return loss, validation_loss

    def _create_network(self, input_size):
        # Create the network
        sequence = [self._fingerprint_scaler]
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

    def _tensor_kwargs(self, **additional) -> dict:
        return dict(device=self._device, dtype=torch.float64, **additional)
