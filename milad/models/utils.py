# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from . import neuralnetwork

__all__ = ('TrainingMonitor',)


class TrainingMonitor:
    """Class that supports monitoring and plotting results of a neural network training procedure"""

    def __init__(self, validation_data: neuralnetwork.FittingData = None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.set_yscale('log')
        plt.ion()

        self.fig = fig
        self.ax = ax
        self._plt_training = None
        self._plt_validation = None

        self.training_maes = []
        self.validation_maes = []

        self._validation_data = validation_data

    def progress_callaback(self, network, step, _training, loss):
        # Calculate MSEs
        training_rmsd = loss.cpu().item()**0.5
        self.training_maes.append(training_rmsd)
        if self._plt_training is not None:
            self._plt_training.remove()
        self._plt_training = self.ax.scatter(list(range(len(self.training_maes))), self.training_maes, c='r', alpha=0.3)

        if self._validation_data:
            # Calculate the validation loss
            validation_rmsd = network.loss(self._validation_data).cpu().item()**0.5

            self.validation_maes.append(validation_rmsd)

            if self._plt_validation is not None:
                self._plt_validation.remove()

            self._plt_validation = self.ax.scatter(
                list(range(len(self.validation_maes))), self.validation_maes, c='b', alpha=0.3
            )
        self.fig.canvas.draw()

    def plot_energy_comparison(
        self, network: neuralnetwork.NeuralNetwork, training_data: neuralnetwork.FittingData,
        *fitting_data: neuralnetwork.FittingData
    ):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()

        minimum, maximum = np.inf, -np.inf

        if training_data:
            energies = network._make_prediction(*training_data.fingerprints)
            known_energies = training_data.get_normalised_energies().cpu().detach().numpy()
            ax.scatter(known_energies, (energies / training_data.num_atoms).cpu().detach().numpy(), c='r', alpha=0.3)
            minimum = min(minimum, known_energies.min())
            maximum = max(maximum, known_energies.max())

        for entry in fitting_data:
            # Any additional datasets the user wants to plot
            energies = network._make_prediction(*entry.fingerprints)
            known_energies = entry.get_normalised_energies().cpu().detach().numpy()
            ax.scatter(known_energies, (energies / entry.num_atoms).cpu().detach().numpy(), alpha=0.4)
            minimum = min(minimum, known_energies.min())
            maximum = max(maximum, known_energies.max())

        if self._validation_data:
            energies = network._make_prediction(*self._validation_data.fingerprints)
            known_energies = self._validation_data.get_normalised_energies().cpu().detach().numpy()
            ax.scatter(
                known_energies, (energies / self._validation_data.num_atoms).cpu().detach().numpy(), c='b', alpha=0.8
            )
            minimum = min(minimum, known_energies.min())
            maximum = max(maximum, known_energies.max())

        if minimum != np.inf:
            ax.plot((minimum, maximum), (minimum, maximum), 'k-', lw=2, c='black')

        return fig

    def plot_energy_deviation_histogram(self, network, *fitting_data: neuralnetwork.FittingData):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.set_ylabel('No. of structures')

        all_datasets = [self._validation_data, *fitting_data]
        for data in all_datasets:
            energies = network._make_prediction(*data.fingerprints, normalise=True)
            differences = data.get_normalised_energies().cpu().detach().numpy() - energies.cpu().detach().numpy()
            ax.hist(differences, 100)

        return fig
