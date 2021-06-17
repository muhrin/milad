# -*- coding: utf-8 -*-
import collections
import math
from typing import Tuple, Optional, Callable

import numpy as np
from scipy import optimize

from milad import functions
from . import utils

__all__ = ('LeastSquaresOptimiser',)

OptimiserResult = collections.namedtuple('OptimiserResult', 'success message value rmsd n_func_eval n_jac_eval')

BoundsType = Tuple[Optional[functions.StateLike], Optional[functions.StateLike]]


class LeastSquaresOptimiser:
    """Uses scipy optimize.least_squares to perform optimisation"""

    def optimise(  # pylint: disable=too-many-locals
            self,
            func: functions.Function,
            initial: functions.StateLike,
            mask: functions.StateLike = None,
            jacobian='native',
            bounds: BoundsType = (-np.inf, np.inf),
            max_func_evals=10000,
            x_tol=1e-8,
            cost_tol=1e-6,
            grad_tol=1e-8,
            callback: Callable = None,
            verbose=False,
    ) -> OptimiserResult:
        """
        :param func: the function to optimise
        :param initial: the initial state
        :param mask: a mask of pre-set values whose degrees of freedom will not be present in the optimisation vector
        :param jacobian: if 'native' the analytic Jacobian will be requested from 'func', otherwise this option is
            passed to optimize.least_squares
        :param bounds: place bounds on the possible inputs to func
        :param max_func_evals: the maximum number of function evaluations.  If not specified will be 100 * len(initial).
        :param x_tol: tolerance for termination by change of the independent variables.
        :param cost_tol: tolerance in change of cost function.  If the difference between one step and another goes
            below this value the optimisation will stop
        :param verbose: print during the optimisation

        :return:
        """
        # SciPy can only deal with numpy arrays so if we get one, all good, otherwise we have to rely on a builder
        if isinstance(initial, np.ndarray):
            fun = func
            x0_ = initial
            builder = None
        else:
            # Need to add a builder step
            builder = initial.get_builder(mask=mask)
            fun = functions.Chain(builder, func)
            x0_ = builder.inverse(initial)
            bounds = list(bounds)
            # Convert the bounds
            if isinstance(bounds[0], type(initial)):
                bounds[0] = builder.inverse(bounds[0])
            if isinstance(bounds[1], type(initial)):
                bounds[1] = builder.inverse(bounds[1])
            bounds = tuple(bounds)

        complex_input = False
        if np.iscomplexobj(x0_):
            complex_input = True
            x0_ = utils.split_state(x0_)

        # Annoyingly scipy least_squares calls the function and the separately
        # but we don't support getting the Jacobian on it's own (it always comes
        # with a function evaluation and we don't want to call it twice so just
        # cache the result)
        jac = self._jac if jacobian == 'native' else jacobian
        data = utils.Data(
            use_jacobian=jacobian == 'native',
            complex_input=complex_input,
            verbose=verbose,
            builder=builder,
            callback=callback,
        )

        max_func_evals = max_func_evals if max_func_evals is not None else 100 * len(initial)

        if np.isnan(x0_).any():
            raise ValueError()

        # Do it!
        res = optimize.least_squares(
            self._calc,
            x0_,
            jac=jac,
            kwargs=dict(func=fun, opt_data=data),
            bounds=bounds,
            xtol=x_tol,
            ftol=cost_tol,
            gtol=grad_tol,
            max_nfev=max_func_evals,
        )

        if complex_input:
            value = utils.join_state(res.x)
        else:
            value = res.x

        if builder is not None:
            value = builder(value)

        # Now convert the result to our optimiser result format
        return OptimiserResult(
            success=res.success,
            value=value,
            rmsd=math.sqrt(2. / len(res.x) * res.cost),
            message=res.message,
            n_func_eval=res.nfev,
            n_jac_eval=res.njev,
        )

    def optimise_target(
        self,
        func: functions.Function,
        initial: functions.State,
        target: functions.StateLike,
        mask: functions.State = None,
        jacobian='native',
        bounds: BoundsType = (-np.inf, np.inf),
        max_func_evals=10000,
        x_tol=1e-8,
        cost_tol=1e-6,
        grad_tol=1e-8,
        callback: Callable = None,
        verbose=False,
    ) -> OptimiserResult:
        """
        Optimise the function to a given target value.  This minimises the loss of the difference between the function
        and the target.

        :param func: the function to optimise
        :param initial: the initial state
        :param target: the output state
        :param jacobian: if 'native' the analytic Jacobian will be requested from 'func', otherwise this option is
            passed to optimize.least_squares
        :param bounds: place bounds on the possible inputs to func
        :param max_func_evals: the maximum number of force evaluations.  If not specified will be 50 * len(initial).
        :param x_tol: tolerance for termination by change of the independent variables.
        :param cost_tol: tolerance in change of cost function.  If the difference between one step and another goes
            below this value the optimisation will stop
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
            x_tol=x_tol,
            cost_tol=cost_tol,
            grad_tol=grad_tol,
            max_func_evals=max_func_evals,
            callback=callback,
            verbose=verbose,
        )

    @staticmethod
    def _calc(state: functions.StateLike, func: functions.Function, opt_data: 'utils.Data') -> np.ndarray:
        if np.isnan(state).any():
            raise ValueError(f'iter: {opt_data.iter}, state: {state}')

        if opt_data.complex_input:
            state = utils.join_state(state)

        value = functions.get_bare(func(state))
        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))

        if opt_data.callback is not None:
            if opt_data.builder is None:
                inp = state
            else:
                inp = opt_data.builder(state)
            opt_data.callback(inp, value, None)

        split = utils.split_state(value)

        if np.isnan(split).any():
            raise ValueError(f'iter: {opt_data.iter}, split: {split}, value: {value}, state: {state}')

        opt_data.iter += 1

        return split

    @staticmethod
    def _jac(state: functions.StateLike, func: functions.Function, opt_data: 'utils.Data') -> np.ndarray:
        if np.isnan(state).any():
            raise ValueError()

        if opt_data.complex_input:
            state = utils.join_state(state)

        value, jac = func(state, jacobian=True)
        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))

        if opt_data.callback is not None:
            if opt_data.builder is None:
                inp = state
            else:
                inp = opt_data.builder(state)
            opt_data.callback(inp, value, jac)

        split = utils.split_jacobian(jac, complex_inputs=opt_data.complex_input, complex_outputs=np.iscomplexobj(value))

        if np.isnan(split).any():
            raise ValueError()

        return split
