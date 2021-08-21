# -*- coding: utf-8 -*-
import argparse
import collections
import functools
from typing import Tuple, Union, List

import numpy as np

from milad import atomic
from milad import base_moments
from milad import fingerprinting
from milad import functions
from milad.play import asetools
from . import least_squares

__all__ = ('StructureOptimiser', 'StructureOptimisationResult')

StructureOptimisationResult = collections.namedtuple(
    'StructureOptimisationResult', 'success message value rmsd n_func_eval n_jac_eval traj'
)


class StructureOptimiser:
    """
    The structure optimiser takes a fingerprinting function and performs a least squares optimisation
    to match a structure to a given fingerprint.
    """

    def __init__(self):
        self._least_squares_optimiser = least_squares.LeastSquaresOptimiser()
        self._separation_force = None

    @property
    def separation_force(self) -> atomic.SeparationForce:
        return self._separation_force

    @separation_force.setter
    def separation_force(self, force):
        self._separation_force = force

    def optimise(
        # pylint: disable=too-many-locals
        self,
        descriptor: fingerprinting.MomentInvariantsDescriptor,
        target: Union[np.ndarray, base_moments.Moments],
        initial: atomic.AtomsCollection,
        jacobian='native',
        mask: atomic.AtomsCollection = None,
        bounds: Tuple[atomic.AtomsCollection, atomic.AtomsCollection] = None,
        x_tol=1e-8,
        cost_tol=1e-7,
        grad_tol=1e-10,
        max_func_evals=5000,
        preprocess=True,
        preprocessor: functions.Function = None,
        get_trajectory=False,
        verbose=False,
    ) -> StructureOptimisationResult:
        """
        :param descriptor: the descriptor with the settings used to generate the fingerprint
        :param target: the fingerprint to decode back into an atoms collection
        :param initial: the starting atoms configuration
        :param x_tol: tolerance for termination by the change of independent variables.
        :param cost_tol: stopping criterion for the fitting algorithm
        :param max_func_evals: the maximum number of allowed fingerprint evaluations
        :param preprocess: if True will apply the descriptor preprocessor to initial and the mask and then
            'un-preprocess' the output
        :param preprocessor: the function used to perform preprocessing.  The following inputs (if present) will be
            preprocessed:
                * initial
                * mask
                * bounds
            At the end of the optimisation result.value = preprocessor.inverse(result.value) will be applied

        :return: a structure optimisation result
        """
        # pylint: disable=too-many-branches
        outcome = argparse.Namespace()

        if isinstance(descriptor, fingerprinting.MomentInvariantsDescriptor):
            # Special case when the function being optimised is a descriptor
            if isinstance(target, base_moments.Moments):
                calc = descriptor.process[:-1]
            elif isinstance(target, np.ndarray):
                calc = descriptor.process
            else:
                raise TypeError(f'Unsupported type {target.__class__.__name__}')

            if preprocess:
                preprocessor = descriptor.preprocess

            bounds = bounds or descriptor.get_bounds(initial.num_atoms)
        else:
            calc = descriptor
            preprocess = False
            if preprocessor is not None:
                preprocess = True

        if preprocess:
            # Perform preprocessing
            initial = preprocessor(initial)
            if mask is not None:
                mask = preprocessor(mask)
            if bounds is not None:
                # Preprocess the bounds and then check if the species number range is restricted by a map
                bounds = tuple(map(preprocessor, bounds))
                if descriptor.species_mapper is not None:
                    # Set the bounds to the range of the mapped range
                    mapped_range = descriptor.species_mapper.mapped_range
                    bounds[0].numbers = mapped_range[0]
                    bounds[1].numbers = mapped_range[1]

        # Deal with saving of trajectory
        save_traj_fn = None
        if get_trajectory:
            outcome.traj = []
            save_traj_fn = functools.partial(self._save_trajectory, outcome.traj, preprocessor if preprocess else None)
        else:
            outcome.traj = None

        calc = self._prepare_optimisation(calc, target)
        result = self._least_squares_optimiser.optimise(
            func=calc,
            initial=initial,
            # target=target,
            mask=mask,
            jacobian=jacobian,
            bounds=bounds,
            max_func_evals=max_func_evals,
            x_tol=x_tol,
            cost_tol=cost_tol,
            grad_tol=grad_tol,
            callback=save_traj_fn,
            verbose=verbose
        )

        outcome.__dict__.update(result._asdict())

        if preprocess:
            # 'Un-preprocess' the output (e.g. map atomic species into integers)
            outcome.value = preprocessor.inverse(result.value)

        return StructureOptimisationResult(**outcome.__dict__)

    def _prepare_optimisation(self, calc: functions.Function, target) -> functions.Function:
        # We want to get as close to the target as possible
        new_calc = functions.Chain(calc, functions.Residuals(target))

        if self._separation_force:
            # Minimise the sum of the separation force and the MSE to the target
            new_calc = functions.Chain(
                functions.Map(new_calc, self._separation_force),
                functions.HStack(),
            )

        return new_calc

    @staticmethod
    def _save_trajectory(trajectory: List, preprocessor, state: atomic.AtomsCollection, _value: np.ndarray, jacobian):
        if jacobian is not None:
            # Don't bother saving on Jacobian calls
            return

        if preprocessor is not None:
            state = preprocessor.inverse(state)

        trajectory.append(asetools.milad2ase(state))
