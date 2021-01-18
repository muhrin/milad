# -*- coding: utf-8 -*-
import math
from typing import Union, Callable, Sequence, Tuple, Optional, List

import ase
import torch
import torch.nn.functional
from torch.autograd import functional

from milad import atomic
from milad.play import asetools
from milad import fingerprinting
from milad import utils

__all__ = 'NeuralNetwork', 'create_fingerprint_set'


def create_fingerprint_set(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    systems: Sequence[ase.Atoms],
    get_derivatives=False
) -> utils.FingerprintSet:
    num_atomic_properties = atomic.AtomsCollection.num_atomic_properties()
    fp_length = descriptor.fingerprint_len

    fingerprints = utils.FingerprintSet(descriptor.fingerprint_len)
    for system in systems:
        fps = []
        fp_derivatives = []
        for env in asetools.extract_environments(system, cutoff=descriptor.cutoff):
            if get_derivatives:
                fingerprint, jacobian = descriptor(asetools.ase2milad(env), jacobian=True)
                # Now, let's deal with the derivatives
                derivatives = jacobian.reshape(fp_length, num_atomic_properties, -1).sum(2)

                fps.append(fingerprint)
                fp_derivatives.append(derivatives)
            else:
                fps.append(descriptor(asetools.ase2milad(env)))

        fingerprints.add_system(system, fps, derivatives=fp_derivatives)

    return fingerprints


class Prediction:
    total_energy: torch.Tensor = None
    forces: torch.Tensor = None


class Predictions:

    def __init__(self, dtype, device):
        self._predictions: List[Prediction] = []
        self._dtype = dtype
        self._device = device

    def __getitem__(self, item):
        return self._predictions[item]

    def append(self, prediction: Prediction):
        self._predictions.append(prediction)

    @property
    def energies(self) -> torch.Tensor:
        """Get total energies as a single tensor preserving gradients"""
        return torch.stack(tuple(prediction.total_energy for prediction in self._predictions))

    @property
    def forces(self) -> Tuple[torch.Tensor]:
        return tuple(prediction.forces for prediction in self._predictions)


class FittingData:
    """Data used during a fitting procedure"""

    @classmethod
    def from_fingerprint_set(cls, fingerprint_set: utils.FingerprintSet, device=None, dtype=None, requires_grad=False):
        tensor_kwargs = dict(dtype=dtype, device=device, requires_grad=requires_grad)
        fingerprints = tuple(torch.tensor(fps, **tensor_kwargs) for fps in fingerprint_set.fingerprints)
        energies = torch.tensor(fingerprint_set.get_potential_energies(normalise=False), **tensor_kwargs)
        forces = tuple(
            torch.tensor(forces, **tensor_kwargs) if forces is not None else None
            for forces in fingerprint_set.get_forces()
        )
        derivatives = tuple(
            torch.tensor(derivatives[:, :, :3], **tensor_kwargs) if derivatives is not None else None
            for derivatives in fingerprint_set.fingerprint_derivatives
        )

        return FittingData(fingerprints, energies, forces=forces, derivatives=derivatives)

    def __init__(
        self,
        fingerprints: Tuple[torch.Tensor],
        energies: torch.Tensor,
        forces: Optional[Tuple[torch.Tensor]] = None,
        derivatives: Optional[Tuple[torch.Tensor]] = None
    ):
        """
        Construct a set of fitting data.

        :param fingerprints: a tuple containing a set of fingerprints for each atom in the structure
        :param energies: the corresponding energy for each structure
        :param forces: the (optional) forces for each atom in the structure
        """
        self._fingerprints = fingerprints
        self._energies = energies
        self._num_atoms = torch.tensor(tuple(len(fps) for fps in fingerprints), device=fingerprints[0].device)
        self._forces = forces
        self._derivatives = derivatives

    def __getitem__(self, item) -> 'FittingData':
        if isinstance(item, slice):
            return FittingData(
                self._fingerprints[item], self._energies[item], self._forces[item], self._derivatives[item]
            )

        fingerprints = (self._fingerprints[item],)
        energies = self._energies[item].reshape(1, 1)
        forces = (self._forces[item],) if self._forces is not None else None
        derivatives = (self._derivatives[item],) if self._derivatives is not None else None
        return FittingData(fingerprints, energies, forces, derivatives=derivatives)

    @property
    def fingerprints(self) -> Tuple[torch.Tensor]:
        """Fingerprints tensors.  One entry per system."""
        return self._fingerprints

    @property
    def energies(self) -> torch.Tensor:
        """Get the known energies"""
        return self._energies

    @property
    def forces(self) -> Optional[Tuple[torch.Tensor]]:
        """Get the known forces if we have them"""
        return self._forces

    @property
    def derivatives(self) -> Optional[Tuple[torch.Tensor]]:
        return self._derivatives

    @property
    def num_atoms(self) -> torch.Tensor:
        """Get the number of atoms in each structure"""
        return self._num_atoms

    def get_normalised_energies(self) -> torch.Tensor:
        return self._energies / self._num_atoms


class LossFunction:

    def get_loss(self, predictions: Predictions, fitting_data: FittingData):
        energy_loss = torch.nn.functional.mse_loss(
            predictions.energies / fitting_data.num_atoms, fitting_data.get_normalised_energies()
        )

        force_loss = None
        if fitting_data.forces is not None:

            for predicted_force, training_force in zip(predictions.forces, fitting_data.forces):
                diff = training_force[0] - predicted_force[0]
                if force_loss is None:
                    force_loss = torch.dot(diff, diff)
                else:
                    force_loss += torch.dot(diff, diff)

            force_loss = force_loss / 3.

        return energy_loss + force_loss


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
        use_forces=False,
    ):
        if self._network is None:
            self._create_network(training_set.fingerprint_len)

        training_data = self.create_fitting_data(training_set, requires_grad=True)

        # Scale the inputs and outputs to match the ranges
        norm_energies = training_data.get_normalised_energies()
        self._energy_scaler.output.expand(norm_energies)

        def stopping_function(epoch, _training, _loss):
            if epoch >= max_epochs:
                return True

            return False  # Don't stop

        optimiser = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        loss = self._train(
            training_data,
            stopping_function,
            optimiser,
            batchsize,
            use_forces=use_forces,
            progress_callback=progress_callback,
        )
        return training_data, loss

    def create_fitting_data(self, fingerprint_set: utils.FingerprintSet, requires_grad=False):
        return FittingData.from_fingerprint_set(
            fingerprint_set,
            device=self._device,
            dtype=self._dtype,
            requires_grad=requires_grad,
        )

    def _train(
        self,
        training: FittingData,
        should_stop: Callable,
        optimiser,
        batchsize,
        use_forces=False,
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

                predictions = self._make_prediction2(batch, get_forces=use_forces)

                optimiser.zero_grad()
                loss = self.loss_function.get_loss(predictions, batch)
                loss.backward(retain_graph=True)
                optimiser.step()

            predictions = self._make_prediction2(training, get_forces=True)
            loss = self.loss_function.get_loss(predictions, training)

            if progress_callback is not None:
                progress_callback(self, epoch, training, loss)

            if should_stop(epoch, training, loss):
                break

            epoch += 1

        return loss

    def loss(self, fitting_data: FittingData):
        predictions = self._make_prediction2(fitting_data, get_forces=True)
        return self.loss_function.get_loss(predictions, fitting_data)

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

    def _make_prediction2(self, fitting_data: FittingData, get_forces=False) -> Predictions:
        # Join all fingerprints as this is much faster going through the network
        all_fingerprints = torch.cat(fitting_data.fingerprints, 0)
        local_energies = self._network(all_fingerprints)

        # Now calculate total energy for each system
        predictions = Predictions(device=self._device, dtype=self._dtype)
        idx = 0
        for system_idx, fps in enumerate(fitting_data.fingerprints):
            prediction = Prediction()

            natoms = fps.shape[0]
            summed = torch.sum(local_energies[idx:idx + natoms])

            prediction.total_energy = summed

            if get_forces:
                forces = []
                for fp, deriv in zip(fps, fitting_data.derivatives[system_idx]):
                    # Get the derivative of the energy wrt to input vector
                    network_deriv = functional.jacobian(self._network, fp)
                    predicted_force = -torch.matmul(network_deriv, deriv)
                    forces.append(predicted_force[0])

                prediction.forces = torch.stack(forces)

            predictions.append(prediction)
            idx += natoms

        return predictions

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
