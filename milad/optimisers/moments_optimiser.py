# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np

from milad import base_moments
from milad import invariants as invs
from milad import utils
from . import least_squares
from . import root

__all__ = ('MomentsOptimiser',)


class MomentsOptimiser:
    """Given a vector of invariants this will attempt to recover the corresponding moments"""

    def __init__(self):
        self._least_squares_optimiser = least_squares.LeastSquaresOptimiser()
        # Use a less accurate tolerance while finding one by one and then crank up
        self._mask_invariant_moments = True
        self._root_finger = root.RootFinder()

    def optimise(  # pylint: disable=too-many-locals
            self,
            invariants_fn: invs.MomentInvariants,
            target: np.ndarray,
            initial: base_moments.Moments,
            jacobian='native',
            bounds=(-np.inf, np.inf),
            target_rmsd=1e-5,
            max_func_evals=5000,
            cost_tol=1e-5,
            grad_tol=1e-8,
            max_retries=2,
            verbose=False,
    ) -> least_squares.OptimiserResult:
        # Copy the start point
        current_moments = copy.deepcopy(initial)
        mask = initial.get_mask()
        mask.array.data[:, :, :] = current_moments.array[:, :, :]

        keep_fixed = {}  # Keep track of the ones we shouldn't unmask
        if self._mask_invariant_moments:
            keep_fixed = self._find_invariant_moments(invariants_fn, target)

        # Unmask 1st order
        mask.array.data[1, :, :] = None

        # Optimise from second order up
        for order in utils.inclusive(2, invariants_fn.max_order):
            indices = invariants_fn.find(functools.partial(max_n_degree, order))
            partial_invariants = invariants_fn[indices]
            partial_target = target[list(indices)]

            # Unmask the moments of this order
            mask.array.data[order, :, :] = None

            # Keep track of the best result for this order
            best_result = None
            for attempt in utils.inclusive(max_retries):
                if verbose:
                    print(f'Attempt {attempt}')

                # Let's mask off the degrees of freedom we know don't change
                self._fix_invariant_moments(mask, keep_fixed)

                result = self._least_squares_optimiser.optimise_target(
                    func=partial_invariants,
                    initial=current_moments,
                    target=partial_target,
                    mask=mask,
                    jacobian=jacobian,
                    bounds=bounds,
                    max_func_evals=128,
                    cost_tol=100 * cost_tol,
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

            # Keep the best results for the next order
            current_moments = best_result.value

        mask = current_moments.get_mask()
        self._fix_invariant_moments(mask, keep_fixed)

        # Now let's do one final optimisation to clean things up
        result = self._least_squares_optimiser.optimise_target(
            func=invariants_fn,
            initial=current_moments,
            target=target,
            mask=mask,
            jacobian=jacobian,
            bounds=bounds,
            max_func_evals=max_func_evals,
            cost_tol=cost_tol,
            grad_tol=grad_tol,
            verbose=verbose
        )

        if verbose:
            print(f'Found solution with RMSD: {result.rmsd}')

        return result

    # def optimise4(  # pylint: disable=too-many-locals
    #         self,
    #         invariants_fn: invs.MomentInvariants,
    #         target: np.ndarray,
    #         initial: base_moments.Moments,
    #         jacobian='native',
    #         bounds=(-np.inf, np.inf),
    #         target_rmsd=1e-5,
    #         max_func_evals=5000,
    #         cost_tol=1e-5,
    #         grad_tol=1e-8,
    #         max_retries=1,
    #         verbose=False,
    # ) -> least_squares.OptimiserResult:
    #     # Copy the start point
    #     current_moments = copy.deepcopy(initial)
    #     mask = initial.get_mask()
    #     mask.array.data[:, :, :] = current_moments.array[:, :, :]
    #
    #     keep_fixed = {}  # Keep track of the ones we shouldn't unmask
    #     if self._mask_invariant_moments:
    #         keep_fixed = self._find_invariant_moments(invariants_fn, target)
    #
    #     # Unmask first two orders
    #     mask.array.data[1:3, :, :] = None
    #
    #     indices = invariants_fn.find(functools.partial(max_n_degree, 2))
    #     partial_invariants = invariants_fn[indices]
    #     partial_target = target[list(indices)]
    #
    #     # Let's mask off the degrees of freedom we know don't change
    #     self._fix_invariant_moments(mask, keep_fixed)
    #
    #     result = self._least_squares_optimiser.optimise_target(
    #         func=partial_invariants,
    #         initial=current_moments,
    #         target=partial_target,
    #         mask=mask,
    #         jacobian=jacobian,
    #         bounds=bounds,
    #         max_func_evals=128,
    #         cost_tol=100 * cost_tol,
    #         grad_tol=100 * grad_tol,
    #         verbose=verbose
    #     )
    #     # Copy over first and second order values
    #     mask.array.data[0:3, :, :] = result.value.array[0:3, :, :]
    #
    #     done_invariants = set(indices)
    #     for order in utils.inclusive(3, invariants_fn.max_order):
    #         if verbose:
    #             print(f'==Doing up to N={order}')
    #
    #         # Unmask this order
    #         mask.array.data[order, :, :] = None
    #         self._fix_invariant_moments(mask, keep_fixed)
    #
    #         this_order = invariants_fn.find(functools.partial(max_n_degree, order))
    #         indices = tuple(set(this_order) - done_invariants)
    #         partial_invariants = invariants_fn[indices]
    #         partial_target = target[list(indices)]
    #
    #         # Now use root finder to solve the rest
    #         result = root.RootFinder().optimise_target(
    #             partial_invariants, result.value, target=partial_target, mask=mask, max_func_evals=500,
    #             verbose=verbose
    #         )
    #
    #         # Copy over found values
    #         mask.array.data[order, :, :] = result.value.array[order, :, :]
    #         done_invariants.update(set(indices))
    #
    #     if verbose:
    #         print('==Doing final relaxation==')
    #
    #     mask = current_moments.get_mask()
    #     self._fix_invariant_moments(mask, keep_fixed)
    #
    #     # Now let's do one final optimisation to clean things up
    #     result = self._least_squares_optimiser.optimise_target(
    #         func=invariants_fn,
    #         initial=result.value,
    #         target=target,
    #         mask=mask,
    #         jacobian=jacobian,
    #         bounds=bounds,
    #         max_func_evals=max_func_evals,
    #         cost_tol=cost_tol,
    #         grad_tol=grad_tol,
    #         verbose=verbose
    #     )
    #
    #     if verbose:
    #         print(f'Found solution with RMSD: {result.rmsd}')
    #
    #     return result

    @staticmethod
    def _find_invariant_moments(invariants_fn: invs.MomentInvariants, invariants: np.ndarray) -> dict:
        # Let's mask off the degrees of freedom we know don't change
        invariant_moments = {}
        for inv_idx, inv in enumerate(invariants_fn):
            # If the invariant is equals just one moment then the moment itself is an invariant so mask it
            if inv.terms.shape == (1, 1, 3):
                idx = tuple(inv.terms[0, 0])
                invariant_moments[idx] = invariants[inv_idx]

        return invariant_moments

    @staticmethod
    def _fix_invariant_moments(moments: base_moments.Moments, invariant_moments: dict):
        # Let's mask off the degrees of freedom we know don't change
        for mom_idx, inv_value in invariant_moments.items():
            if max(mom_idx) > moments.max_order:
                continue
            moments[mom_idx] = inv_value


def max_n_degree(max_n: int, invariant: invs.MomentInvariant):
    return invariant.terms_array[:, :, 0].max() <= max_n


def n_degree(  # pylint: disable=invalid-name
    n: int, invariant: invs.MomentInvariant
):
    return np.all(invariant.terms_array[:, :, 0] == n)


def max_l_degree(max_l: int, invariant: invs.MomentInvariant):
    # The indexing is n, l, m so l is at index 1
    return invariant.terms_array[:, :, 1].max() <= max_l


def l_degree(  # pylint: disable=invalid-name
    l: int, invariant: invs.MomentInvariant
):
    # The indexing is n, l, m so l is at index 1
    return np.all(invariant.terms_array[:, :, 1] == l)


def containing_l_degree(  # pylint: disable=invalid-name
    l: int, invariant: invs.MomentInvariant
):
    return np.any(invariant.terms_array[:, :, 1] == l) and invariant.terms_array[:, :, 1].max() <= l
