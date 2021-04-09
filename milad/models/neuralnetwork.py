# -*- coding: utf-8 -*-
import collections
import math
from typing import Union, Callable, Sequence, Optional, List

import ase
import torch
import torch.nn.functional
from torch.autograd import functional

from milad.play import asetools
from milad import fingerprinting
from milad import utils

__all__ = 'NeuralNetwork', 'create_fingerprint_set'

# pylint: disable=no-member, not-callable


def create_fingerprint_set(
    descriptor: fingerprinting.MomentInvariantsDescriptor,
    systems: Sequence[ase.Atoms],
    get_derivatives=False
) -> utils.FingerprintSet:
    # WARNING: The calculation of derivatives only takes into account positional degrees of freedom (not species) and
    # makes assumptions about the shape of the derivatives tensor

    fp_length = descriptor.fingerprint_len

    fingerprints = utils.FingerprintSet(descriptor.fingerprint_len)
    for system in systems:
        fps = []
        fp_derivatives = []
        for env in asetools.extract_environments(system, cutoff=descriptor.cutoff):
            milad_env = asetools.ase2milad(env)
            if get_derivatives:
                fingerprint, jacobian = descriptor(milad_env, jacobian=True)
                # Now, let's deal with the derivatives extracting just the positional parts and summing over all
                # neighbours keeping x, y, z separate
                derivatives = jacobian[:, 3:3 * len(env)].reshape(fp_length, -1, 3).sum(1)

                fps.append(fingerprint)
                fp_derivatives.append(derivatives)
            else:
                fps.append(descriptor(milad_env))

        fingerprints.add_system(system, fps, derivatives=fp_derivatives)

    return fingerprints


class Predictions:

    def __init__(self, sizes: torch.Tensor, local_energies: torch.Tensor, forces: torch.Tensor = None):
        """Create an object to store prediction data from the neural network

        :param sizes: a tensor containing the number of atoms in each system.  The length of the tensor is the total
            number of systems the prediction is being made for.
        """
        if not sizes.sum() == len(local_energies):
            raise ValueError(
                f"The total number of atoms ({sizes.sum()}) doesn't match the number of "
                f'local energies ({len(local_energies)}).'
            )

        self.sizes = sizes
        self.local_energies = local_energies
        self.forces = forces
        self._total_energies = None

    @property
    def total_energies(self) -> torch.Tensor:
        """Get total energies as a single tensor that preserves gradients"""
        if self._total_energies is None:
            # Lazily calculate
            idx = 0
            total_energies = []
            for num_atoms in self.sizes:
                total_energies.append(torch.sum(self.local_energies[idx:idx + num_atoms]))
                idx += num_atoms
            # Combine them all into a tensor
            self._total_energies = torch.stack(total_energies)

        return self._total_energies

    def get_normalised_energies(self) -> torch.Tensor:
        """Get the energy/atom for each system"""
        return self.total_energies / self.sizes


class FittingData:
    """Data used during a fitting procedure"""

    @classmethod
    def from_fingerprint_set(cls, fingerprint_set: utils.FingerprintSet, device=None, dtype=None, requires_grad=False):
        tensor_kwargs = dict(dtype=dtype, device=device, requires_grad=requires_grad)
        fingerprints = torch.cat(tuple(torch.tensor(fps, **tensor_kwargs) for fps in fingerprint_set.fingerprints))

        forces = None
        if fingerprint_set.has_all_forces():
            forces = torch.cat(
                tuple(
                    torch.tensor(forces, **tensor_kwargs) if forces is not None else None
                    for forces in fingerprint_set.get_forces()
                )
            )

        derivatives = None
        if fingerprint_set.has_all_derivatives():
            derivatives = torch.cat(
                tuple(
                    torch.tensor(derivatives, **tensor_kwargs) if derivatives is not None else None
                    for derivatives in fingerprint_set.fingerprint_derivatives
                )
            )

        return FittingData(
            torch.tensor(fingerprint_set.sizes, dtype=torch.int, device=device),
            fingerprints,
            total_energies=torch.tensor(fingerprint_set.get_potential_energies(normalise=False), **tensor_kwargs),
            forces=forces,
            derivatives=derivatives
        )

    def __init__(
        self,
        sizes: torch.Tensor,
        fingerprints: torch.Tensor,
        total_energies: torch.Tensor,
        forces: Optional[torch.Tensor] = None,
        derivatives: Optional[torch.Tensor] = None
    ):
        """
        Construct a set of fitting data.

        :param fingerprints: a tuple containing a set of fingerprints for each atom in the structure
        :param total_energies: the corresponding energy for each structure
        :param forces: the (optional) forces for each atom in the structure
        """
        if sizes.sum() != len(fingerprints):
            raise ValueError(
                f"The total number of atoms ({sizes.sum()}) doesn't match the number of "
                f'fingerprints ({len(fingerprints)}).'
            )
        if len(sizes) != len(total_energies):
            raise ValueError(
                f"The number of total energies ({len(total_energies)}) doesn't match the number of "
                f'systems (sizes) ({len(sizes)}).'
            )

        if not (len(derivatives) == len(fingerprints) == len(forces)):  # pylint: disable=superfluous-parens
            raise ValueError('Dataset size mismatch')

        self._num_atoms = sizes
        self._fingerprints = fingerprints
        self._total_energies = total_energies
        self._forces = forces
        self._derivatives = derivatives

    def __getitem__(self, item) -> 'FittingData':
        if isinstance(item, slice):
            atomic_slice = self._get_per_atom_slice(item)

            sizes = self._num_atoms[item]
            total_energies = self._total_energies[item]
            fingerprints = self._fingerprints[atomic_slice]
            forces = self._forces[atomic_slice] if self._forces is not None else None
            derivatives = self._derivatives[atomic_slice] if self._derivatives is not None else None

            return FittingData(sizes, fingerprints, total_energies, forces, derivatives)

        # Assume item is a fixed index
        return self.__getitem__(slice(item, item + 1))

    def __len__(self) -> int:
        """Returns the total number of systems contained in this fitting data"""
        return len(self._num_atoms)

    def _get_per_atom_slice(self, item: slice) -> slice:
        if item.step is not None and item.step != 1:
            raise ValueError(f'Non-contigious slices are unsupported for now, got step {item.step}')

        start = self._num_atoms[:item.start].sum().cpu().item()
        stop = self._num_atoms[:item.stop].sum().cpu().item()
        return slice(start, stop)

    @property
    def fingerprints(self) -> torch.Tensor:
        """Fingerprints tensors.  One entry per system."""
        return self._fingerprints

    @property
    def total_energies(self) -> torch.Tensor:
        """Get the known energies"""
        return self._total_energies

    @property
    def forces(self) -> Optional[torch.Tensor]:
        """Get the known forces if we have them"""
        return self._forces

    @property
    def derivatives(self) -> Optional[torch.Tensor]:
        return self._derivatives

    @property
    def num_atoms(self) -> torch.Tensor:
        """Get the number of atoms in each structure"""
        return self._num_atoms

    def get_normalised_energies(self) -> torch.Tensor:
        return self._total_energies / self._num_atoms

    def batch_split(self, batchsize: int) -> List['FittingData']:
        total_samples = len(self)
        num_batches = int(math.ceil(total_samples / batchsize))
        batches = []
        for batch_num in range(num_batches):
            start_idx = batch_num * batchsize
            end_idx = min(start_idx + batchsize, total_samples)
            batches.append(self[start_idx:end_idx])

        return batches


class LossFunction:
    Loss = collections.namedtuple('Result', 'energy force total')

    def __init__(self, energy_coeff=1., force_coeff=0.1):
        self.energy_coeff = energy_coeff
        self.force_coeff = force_coeff

    def get_loss(self, predictions: Predictions, fitting_data: FittingData):
        total_loss = 0.
        energy_loss = torch.nn.functional.mse_loss(
            predictions.get_normalised_energies(), fitting_data.get_normalised_energies()
        )
        total_loss += self.energy_coeff * energy_loss

        force_loss = None
        if fitting_data.forces is not None and self.force_coeff != 0.:
            force_loss = torch.nn.functional.mse_loss(predictions.forces, fitting_data.forces) / 3.
            total_loss += self.force_coeff * force_loss
            total_loss *= 0.5

        return LossFunction.Loss(energy_loss, force_loss, total_loss)


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
        dtype=torch.float64,
        bias=False,
        vectorise_jacobian=False,
    ):
        """

        :param hiddenlayers: a tuple of the hidden layers to use e.g. (32, 16, 8)
        :param activations:
        :param loss_function:
        :param device:
        :param dtype:
        :param bias:
        :param vectorise_jacobian: if True, will calculate the NN Jacobian for a batch of fingerprints all in one go.
            This is very memory consuming at the moment and may exceed the amount of memory you have available.
        """
        # Set up the hidden layers
        self._hidden = hiddenlayers

        # Set the activation function
        if isinstance(activations, str):
            self._activations = getattr(torch.nn, activations)()
        elif isinstance(activations, Callable):  # pylint: disable=isinstance-second-argument-not-valid-type
            self._activations = activations
        else:
            raise TypeError('Expecting str or Callable, got {}'.format(activations.__class__.__name__))

        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device
        self._dtype = dtype
        self._bias = bias
        self._vectorise_jacobian = vectorise_jacobian

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
        batchsize: int,
        progress_callback: Callable = None,
    ) -> LossFunction.Loss:
        # Let's break up the training data into batches
        batches = training.batch_split(batchsize)

        epoch = 0
        loss_result = None
        get_forces = self.loss_function.force_coeff != 0.
        while True:
            for batch in batches:
                predictions = self.make_prediction(batch, get_forces=get_forces, create_graph=True)

                optimiser.zero_grad()
                loss_result = self.loss_function.get_loss(predictions, batch)
                loss = loss_result.total
                loss.backward(retain_graph=True)
                optimiser.step()

            # Use no_grad to reduce memory footprint
            with torch.no_grad():
                # Calculate loss for the entire training set.  Only necessary if there is more than one batch
                if len(batches) > 1:
                    predictions = self.make_prediction(training, get_forces=get_forces)
                    loss_result = self.loss_function.get_loss(predictions, training)

                if progress_callback is not None:
                    progress_callback(self, epoch, training, loss_result)

                if should_stop(epoch, training, loss_result):
                    break

            epoch += 1

        return loss_result

    def loss(self, fitting_data: FittingData, get_forces=None) -> LossFunction.Loss:
        get_forces = get_forces if get_forces is not None else self.loss_function.force_coeff != 0.
        predictions = self.make_prediction(fitting_data, get_forces=get_forces)
        return self.loss_function.get_loss(predictions, fitting_data)

    def make_prediction(self, fitting_data: FittingData, get_forces=False, create_graph=False) -> Predictions:
        # Join all fingerprints as this is much faster going through the network
        local_energies = self._network(fitting_data.fingerprints)

        forces = None
        if get_forces:
            forces = []
            # Need to calculate the forces using the chain rule with the fingerprints derivative and the
            # neural network derivative
            if self._vectorise_jacobian:
                jac = functional.jacobian(
                    self._network, fitting_data.fingerprints, create_graph=create_graph, vectorize=True
                )
                jacs = [jac[i, :, i, :] for i in range(len(fitting_data.fingerprints))]

                for fingerprint, derivatives, network_deriv in zip(
                    fitting_data.fingerprints, fitting_data.derivatives, jacs
                ):
                    # Get the derivative of the energy wrt to input vector
                    predicted_force = -torch.matmul(network_deriv, derivatives)
                    forces.append(predicted_force)
            else:
                # Fallback, lower memory version
                for fingerprint, derivatives in zip(fitting_data.fingerprints, fitting_data.derivatives):
                    # Get the derivative of the energy wrt to input vector
                    network_deriv = functional.jacobian(
                        self._network, fingerprint, create_graph=create_graph, vectorize=True
                    )
                    predicted_force = -torch.matmul(network_deriv, derivatives)
                    forces.append(predicted_force)

            # VMAP version, doesn't work yet but may be fixed soon and would almost certainly be fast and relatively
            # low-memory.  See:
            # https://github.com/pytorch/pytorch/issues/42368
            # def get_jacobian(fingerprint):
            #     return functional.jacobian(self._network, fingerprint, create_graph=False)
            #
            # jac = torch.vmap(get_jacobian)(tuple(fitting_data.fingerprints))
            # jacs = [jac[i, :, i, :] for i in range(len(fitting_data.fingerprints))]
            #
            # for fingerprint, derivatives, network_deriv
            #   in zip(fitting_data.fingerprints, fitting_data.derivatives, jacs):
            #     # Get the derivative of the energy wrt to input vector
            #     predicted_force = -torch.matmul(network_deriv, derivatives)
            #     forces.append(predicted_force)

            forces = torch.cat(forces)

        return Predictions(fitting_data.num_atoms, local_energies=local_energies, forces=forces)

    def _create_network(self, input_size):
        # Create the network
        # sequence = [self._fingerprint_scaler]
        sequence = []
        prev_size = input_size
        # Create the fully connected (hidden) layers
        for size in self._hidden:
            sequence.append(torch.nn.Linear(prev_size, size, bias=self._bias))
            sequence.append(self._activations)
            prev_size = size
        # Add the output layer
        sequence.extend((torch.nn.Linear(prev_size, 1, bias=self._bias), self._energy_scaler))

        self._network = torch.nn.Sequential(*sequence)
        self._network.to(self._device, self._dtype)

    def tensor(self, *args, requires_grad=False) -> torch.Tensor:
        return torch.tensor(*args, device=self._device, dtype=self._dtype, requires_grad=requires_grad)
