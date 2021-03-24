# -*- coding: utf-8 -*-
import argparse
import collections
import functools
from typing import Tuple, Union, List

import numpy as np

from milad import atomic
from milad import base_moments
from milad import fingerprinting
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
        cost_tol=1e-5,
        grad_tol=1e-8,
        max_func_evals=5000,
        preprocess=True,
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
        :return: a structure optimisation result
        """
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
                initial = preprocessor(initial)
                if mask is not None:
                    mask = preprocessor(mask)

            bounds = bounds or self.get_bounds(initial.num_atoms, descriptor)

        else:
            calc = descriptor
            preprocess = False

        # Deal with saving of trajectory
        save_traj_fn = None
        if get_trajectory:
            outcome.traj = []
            save_traj_fn = functools.partial(self._save_trajectory, outcome.traj, preprocessor if preprocess else None)
        else:
            outcome.traj = None

        result = self._least_squares_optimiser.optimise_target(
            func=calc,
            initial=initial,
            target=target,
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
            # 'Un-preprocess' the output (map atomic species into integers)
            outcome.value = descriptor.preprocess.inverse(result.value)

        return StructureOptimisationResult(**outcome.__dict__)

    @staticmethod
    def _save_trajectory(trajectory: List, preprocessor, state: atomic.AtomsCollection, _value: np.ndarray, jacobian):
        if jacobian is not None:
            # Don't bother saving on Jacobian calls
            return

        if preprocessor is not None:
            state = preprocessor.inverse(state)

        trajectory.append(asetools.milad2ase(state))

    @staticmethod
    def get_bounds(num_atoms: int,
                   descriptor: fingerprinting.MomentInvariantsDescriptor) \
            -> Tuple[atomic.AtomsCollection, atomic.AtomsCollection]:
        species_range = None
        cutoff = descriptor.cutoff

        # Let's look inside the preprocessing step to see if we're mapping atomic numbers, in which case we can
        # use this to set bounds on the species
        results = descriptor.preprocess.find_type(atomic.MapNumbers)
        if results:
            species_range = results[0][1].mapped_range

        return StructureOptimiser.create_bounds(num_atoms, cutoff, species_range)

    @staticmethod
    def create_bounds(num_atoms: int, cutoff=None, species_range=None) ->\
            Tuple[atomic.AtomsCollection, atomic.AtomsCollection]:
        """Create optimisation bounds that optionally restrict the range cartesian values the atomic positions can take
        and the range of atomic species allowed."""
        lower = atomic.AtomsCollection(num_atoms)
        upper = atomic.AtomsCollection(num_atoms)
        lower.vector[:] = -np.inf
        upper.vector[:] = np.inf

        if species_range:
            lower.numbers = species_range[0]
            upper.numbers = species_range[1]

        if cutoff:
            lower.positions = -cutoff
            upper.positions = cutoff

        return lower, upper
