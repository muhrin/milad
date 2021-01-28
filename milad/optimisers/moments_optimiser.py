# -*- coding: utf-8 -*-
import copy

import numpy as np

from milad import base_moments
from milad import invariants as invs
from milad import utils
from . import least_squares

__all__ = ('MomentsOptimiser',)


class MomentsOptimiser:
    """Given a vector of invariants this will attempt to recover the corresponding moments"""

    def __init__(self):
        self._least_squares_optimiser = least_squares.LeastSquaresOptimiser()
        # Use a less accurate tolerance while finding one by one and then crank up
        self._mask_invariant_moments = True

    def optimise(# pylint: disable=too-many-locals
        self,
        invariants_fn: invs.MomentInvariants,
        target: np.ndarray,
        initial: base_moments.Moments,
        jacobian='native',
        bounds=(-np.inf, np.inf),
        target_rmsd=1e-5,
        max_force_evals=1000,
        func_tol=1e-7,
        grad_tol=1e-7,
        max_retries=5,
        verbose=False,
    ) -> least_squares.OptimiserResult:
        # Copy the start point
        current_moments = copy.deepcopy(initial)
        mask = initial.get_mask(fill='self')

        keep_fixed = {}  # Keep track of the ones we shouldn't unmask
        if self._mask_invariant_moments:
            keep_fixed = self._find_invariant_moments(invariants_fn, target)

        for order in utils.inclusive(1, invariants_fn.max_order):
            indices = invariants_fn.find_up_to(order)
            partial_invariants = invariants_fn[indices]
            partial_target = target[list(indices)]

            # Unmask the moments of this order
            mask[order, None, None] = None
            # Let's mask off the degrees of freedom we know don't change
            self._fix_invariant_moments(mask, keep_fixed)

            # Keep track of the best result for this order
            retries = 0
            best_result = None
            for _ in range(max_retries):
                result = self._least_squares_optimiser.optimise_target(
                    func=partial_invariants,
                    initial=current_moments,
                    target=partial_target,
                    mask=mask,
                    jacobian=jacobian,
                    bounds=bounds,
                    max_force_evals=128,
                    cost_tol=100 * func_tol,
                    grad_tol=100 * grad_tol,
                    verbose=verbose
                )

                if verbose:
                    print(f'Found l={order} with RMSD: {result.rmsd}')

                # Keep track of the best result so far
                if best_result is None or result.rmsd < best_result.rmsd:
                    best_result = result

                if best_result.rmsd <= 100 * target_rmsd:
                    if verbose:
                        print('Keeping')
                    break

                # Going to have to try again: generate new moments of this order
                current_moments.randomise(indices=(order, None, None))
                retries += 1
                if verbose:
                    print('Retrying')

            # Keep the best results for the next order
            current_moments = best_result.value

        # Now let's do one final optimisation to clean things up
        result = self._least_squares_optimiser.optimise_target(
            func=invariants_fn,
            initial=current_moments,
            target=target,
            mask=mask,
            jacobian=jacobian,
            bounds=bounds,
            max_force_evals=max_force_evals,
            cost_tol=func_tol,
            grad_tol=grad_tol,
            verbose=verbose
        )

        if verbose:
            print(f'Found solution with RMSD: {result.rmsd}')

        return result

    @staticmethod
    def _find_invariant_moments(invariants_fn: invs.MomentInvariants, invariants: np.ndarray) -> dict:
        # Let's mask off the degrees of freedom we know don't change
        invariant_moments = {}
        for inv_idx, inv in enumerate(invariants_fn):
            # If the invariant is equals just one moment then the moment itself is an invariant so mask it
            if len(inv.terms) == 1:
                idx = inv.terms[0][1][0]
                invariant_moments[idx] = invariants[inv_idx]

        return invariant_moments

    @staticmethod
    def _fix_invariant_moments(moments: base_moments.Moments, invariant_moments: dict):
        # Let's mask off the degrees of freedom we know don't change
        for mom_idx, inv_value in invariant_moments.items():
            if mom_idx[0] > moments.max_order:
                continue
            moments[mom_idx] = inv_value
