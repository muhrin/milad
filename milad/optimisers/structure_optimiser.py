# -*- coding: utf-8 -*-
import collections
from typing import Tuple, Union

import numpy as np

from milad import atomic
from milad import base_moments
from milad import fingerprinting
from . import least_squares

__all__ = ('StructureOptimiser',)

StructureOptimisationResult = collections.namedtuple('StructureOptimisationResult', 'success atoms message rmsd')


class StructureOptimiser:
    """
    The structure optimiser takes a fingerprinting function and performs a least squares optimistaion
    to match a structure to a given fingerprint.
    """

    def __init__(self):
        self._least_squares_optimiser = least_squares.LeastSquaresOptimiser()

    def optimise(
        self,
        descriptor: fingerprinting.MomentInvariantsDescriptor,
        target: Union[np.ndarray, base_moments.Moments],
        initial: atomic.AtomsCollection,
        jacobian='native',
        mask: atomic.AtomsCollection = None,
        x_tol=1e-8,
        cost_tol=1e-5,
        grad_tol=1e-8,
        max_func_evals=5000,
        verbose=False,
    ) -> StructureOptimisationResult:
        """
        :param descriptor: the descriptor with the settings used to generate the fingerprint
        :param target: the fingerprint to decode back into an atoms collection
        :param initial: the starting atoms configuration
        :param x_tol: tolerance for termination by the change of independent variables.
        :param cost_tol: stopping criterion for the fitting algorithm
        :param max_func_evals: the maximum number of allowed fingerprint evaluations
        :param atoms_builder: an optional atoms builder that can be used to freeze certain degrees of freedom
        :return: a structure optimisation result
        """
        if isinstance(target, base_moments.Moments):
            calc = descriptor.process[:-1]
        elif isinstance(target, np.ndarray):
            calc = descriptor.process
        else:
            raise TypeError(f'Unsupported type {target.__class__.__name__}')

        preprocess = descriptor.preprocess
        preprocessed = preprocess(initial)
        if mask is not None:
            mask = preprocess(mask)

        result = self._least_squares_optimiser.optimise_target(
            func=calc,
            initial=preprocessed,
            target=target,
            mask=mask,
            jacobian=jacobian,
            bounds=self._get_bounds(initial.num_atoms, descriptor),
            max_func_evals=max_func_evals,
            x_tol=x_tol,
            cost_tol=cost_tol,
            grad_tol=grad_tol,
            verbose=verbose
        )
        # Build the atoms from the vector and then 'un-preprocess' it (likely scale size and map species)
        return result._replace(value=preprocess.inverse(result.value))

    @staticmethod
    def _get_bounds(num_atoms: int,
                    descriptor: fingerprinting.MomentInvariantsDescriptor) \
            -> Tuple[atomic.AtomsCollection, atomic.AtomsCollection]:
        lower = atomic.AtomsCollection(num_atoms)
        upper = atomic.AtomsCollection(num_atoms)
        lower.vector[:] = -np.inf
        upper.vector[:] = np.inf

        # Let's look inside the preprocessing step to see if we're mapping atomic numbers, in which case we can
        # use this to set bounds on the species
        results = descriptor.preprocess.find_type(atomic.MapNumbers)
        if results:
            species_range = results[0][1].mapped_range
            lower.numbers = species_range[0]
            upper.numbers = species_range[1]

        if descriptor.cutoff is not None:
            lower.positions = -descriptor.cutoff
            upper.positions = descriptor.cutoff

        return lower, upper
