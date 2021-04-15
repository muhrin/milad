# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

import torch

from . import neuralnetwork

__all__ = ('TrainingMonitor',)


class TrainingMonitor:
    """Class that supports monitoring and plotting results of a neural network training procedure"""

    class PlottingDataset:

        def __init__(self, axes, scatter_kwargs: dict = None):
            self._axes = axes
            self._last_plt = None
            self._values = []
            self._steps = []
            self._scatter_kwargs = scatter_kwargs or {}

        def append(self, value, step: int = None):
            self._values.append(value)
            self._steps.append(step if step is not None else len(self._steps))

        def replot(self):
            if self._last_plt is not None:
                self._last_plt.remove()
            self._last_plt = self._axes.scatter(self._steps, self._values, **self._scatter_kwargs)

    def __init__(self, validation_data: neuralnetwork.FittingData = None, show_validation_every=10):
        self._show_validation_every = show_validation_every

        fig = plt.figure(figsize=(12, 10))
        energy_axis = fig.add_subplot(111)
        energy_axis.set_yscale('log')
        plt.ion()

        self.fig = fig
        self.energy_axis = energy_axis

        self._energy_training = TrainingMonitor.PlottingDataset(
            self.energy_axis,
            scatter_kwargs=dict(
                edgecolors='tab:orange', alpha=0.6, marker='o', facecolors='none', label='Energy (training)'
            )
        )
        self._energy_validation = TrainingMonitor.PlottingDataset(
            self.energy_axis, scatter_kwargs=dict(c='tab:orange', alpha=0.6, label='Energy (validation)')
        )

        forces_axis = energy_axis.twinx()
        forces_axis.set_yscale('log')

        self._force_training = TrainingMonitor.PlottingDataset(
            forces_axis,
            scatter_kwargs=dict(edgecolors='tab:blue', alpha=0.6, facecolors='none', label='Force (training)')
        )
        self._force_validation = TrainingMonitor.PlottingDataset(
            forces_axis, scatter_kwargs=dict(c='tab:blue', alpha=0.6, label='Force (validation)')
        )

        self._validation_data = validation_data

    @torch.no_grad()
    def progress_callaback(self, network, step, _training, loss):
        # Calculate MSEs
        training_rmsd = loss.energy.cpu().item()**0.5
        self._energy_training.append(training_rmsd)
        self._energy_training.replot()

        if loss.force is not None:
            self._force_training.append(loss.force.cpu().item()**0.5)
            self._force_training.replot()

        if self._validation_data and (step % self._show_validation_every) == 0:
            # Calculate the validation loss
            validation_loss = network.loss(self._validation_data)
            validation_rmsd = validation_loss.energy.cpu().item()**0.5
            self._energy_validation.append(validation_rmsd, step)
            self._energy_validation.replot()

            if validation_loss.force is not None:
                force_rmsd = validation_loss.force.cpu().item()**0.5
                self._force_validation.append(force_rmsd, step)
                self._force_validation.replot()

        self.fig.canvas.draw()

    def plot_energy_comparison(self, network: neuralnetwork.NeuralNetwork, *fitting_data: neuralnetwork.FittingData):
        fig = plt.figure(figsize=(8, 8))
        axis = fig.gca()
        axis.ticklabel_format(useOffset=False)

        minimum, maximum = np.inf, -np.inf

        training_data = None if not fitting_data else fitting_data[0]

        if training_data:
            predictions = network.make_prediction(training_data)
            known_energies = training_data.get_normalised_energies().cpu().detach().numpy()
            axis.scatter(known_energies, predictions.get_normalised_energies().cpu().detach().numpy(), c='r', alpha=0.3)
            minimum = min(minimum, known_energies.min())
            maximum = max(maximum, known_energies.max())

        for entry in fitting_data:
            # Any additional datasets the user wants to plot
            predictions = network.make_prediction(entry)
            known_energies = entry.get_normalised_energies().cpu().detach().numpy()
            axis.scatter(known_energies, predictions.get_normalised_energies().cpu().detach().numpy(), alpha=0.4)
            minimum = min(minimum, known_energies.min())
            maximum = max(maximum, known_energies.max())

        if self._validation_data:
            predictions = network.make_prediction(self._validation_data)
            known_energies = self._validation_data.get_normalised_energies().cpu().detach().numpy()
            axis.scatter(known_energies, predictions.get_normalised_energies().cpu().detach().numpy(), c='b', alpha=0.8)
            minimum = min(minimum, known_energies.min())
            maximum = max(maximum, known_energies.max())

        if minimum != np.inf:
            axis.plot((minimum, maximum), (minimum, maximum), 'k-', lw=2, c='black')

        return fig

    def plot_energy_deviation_histogram(self, network, *fitting_data: neuralnetwork.FittingData, bins=100):
        fig = plt.figure(figsize=(10, 10))
        axis = fig.gca()
        axis.set_ylabel('No. of structures')

        all_datasets = [self._validation_data, *fitting_data]
        for data_set in all_datasets:
            target_energies = data_set.get_normalised_energies().cpu().detach().numpy()
            predicted_energies = network.make_prediction(data_set).get_normalised_energies().cpu().detach().numpy()
            differences = target_energies - predicted_energies
            axis.hist(differences, bins)

        return fig
