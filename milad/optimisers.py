# -*- coding: utf-8 -*-
import collections
import math

import numpy as np
from scipy import optimize

from . import functions

__all__ = ('LeastSquaresOptimiser',)

OptimiserResult = collections.namedtuple('OptimiserResult', 'success message value rmsd')


class LeastSquaresOptimiser:
    """Uses scipy optimize.least_squares to perform optimistaion"""

    class Data:
        """Data used during an optimisation"""

        def __init__(self, use_jacobian=False, verbose=False):
            self.use_jacobian = use_jacobian
            self.verbose = verbose
            self.last_state = None  # The last state that we evaluated the function at
            self.last_jacobian = None

    def __init__(self):
        pass

    def optimise(
        self,
        func: functions.Function,
        initial: functions.State,
        mask: functions.State = None,
        jacobian='2-point',
        bounds=(-np.inf, np.inf),
        max_force_evals=None,
        grad_tol=1e-8,
        verbose=False,
    ) -> OptimiserResult:
        """
        :param func: the function to optimise
        :param initial: the initial state
        :param mask: a mask of pre-set values whose degrees of freedom will not be present in the optimistaion vector
        :param jacobian: if 'native' the analytic Jacobian will be requested from 'func', otherwise this option is
            passed to optimize.least_squares
        :param bounds: place bounds on the possible inputs to func
        :param max_force_evals: the maximum number of force evaluations.  If not specified will be 50 * len(initial).
        :param verbose: print during the optimisation

        :return:
        """
        # SciPy can only deal with numpy arrays so if we get one, all good, otherwise we have to rely on a builder
        if isinstance(initial, np.ndarray):
            fun = func
            x0_ = initial
        else:
            # Need to add a builder step
            builder = initial.get_builder(mask=mask)
            fun = functions.Chain(builder, func)
            x0_ = builder.inverse(initial)

        # Annoyingly scipy least_squares calls the function and the separately
        # but we don't support getting the Jacobian on it's own (it always comes
        # with a function evaluation and we don't want to call it twice so just
        # cache the result)
        jac = self._jac if jacobian == 'native' else jacobian
        data = LeastSquaresOptimiser.Data(use_jacobian=jacobian == 'native', verbose=verbose)

        max_force_evals = max_force_evals if max_force_evals is not None else 100 * len(initial)

        # Do it!
        res = optimize.least_squares(
            self._calc,
            x0_,
            jac=jac,
            kwargs=dict(func=fun, opt_data=data),
            bounds=bounds,
            gtol=grad_tol,
            max_nfev=max_force_evals,
        )

        # Now convert the result to our optimiser result format
        return OptimiserResult(
            success=res.success,
            value=builder(res.x),
            rmsd=math.sqrt(2. / len(x0_) * res.cost),
            message=res.message,
        )

    def optimise_target(
        self,
        func: functions.Function,
        initial: functions.State,
        target: functions.StateLike,
        mask: functions.State = None,
        jacobian='2-point',
        bounds=(-np.inf, np.inf),
        max_force_evals=None,
        grad_tol=1e-8,
        verbose=False,
    ):
        """
        Optimise the function to a given target value.  This minimises the loss of the difference between the function
        and the target.

        :param func: the function to optimise
        :param initial: the initial state
        :param target: the output state
        :param jacobian: if 'native' the analytic Jacobian will be requested from 'func', otherwise this option is
            passed to optimize.least_squares
        :param bounds: place bounds on the possible inputs to func
        :param max_force_evals: the maximum number of force evaluations.  If not specified will be 50 * len(initial).
        :param verbose: print during the optimisation
        :return:
        """
        # Calculate residuals to a particular target
        return self.optimise(
            functions.Chain(func, functions.Residuals(target)),
            initial=initial,
            mask=mask,
            jacobian=jacobian,
            bounds=bounds,
            grad_tol=grad_tol,
            max_force_evals=max_force_evals,
            verbose=verbose,
        )

    def _calc(
        self, state: functions.StateLike, func: functions.Function, opt_data: 'LeastSquaresOptimiser.Data'
    ) -> np.ndarray:
        value = func(state)
        opt_data.last_state = state
        value = value.real
        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))
        return value

    def _jac(
        self, state: functions.StateLike, func: functions.Function, opt_data: 'LeastSquaresOptimiser.Data'
    ) -> np.ndarray:
        _, jac = func(state, jacobian=True)
        opt_data.last_jacobian = jac.real
        return opt_data.last_jacobian
