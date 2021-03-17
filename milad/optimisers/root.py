# -*- coding: utf-8 -*-
import collections
import functools
import math
from typing import Tuple, Optional, Callable

import numpy as np
from scipy import optimize

from milad import functions
from . import utils

__all__ = ('RootFinder',)

OptimiserResult = collections.namedtuple('OptimiserResult', 'success message value rmsd n_func_eval n_jac_eval')

BoundsType = Tuple[Optional[functions.StateLike], Optional[functions.StateLike]]


class RootFinder:
    """Uses scipy optimize.least_squares to perform optimisation"""

    class Data:
        """Data used during an optimisation"""

        def __init__(self, use_jacobian, complex_input: bool, verbose=False, builder=None, callback: Callable = None):
            self.use_jacobian = use_jacobian
            self.complex_input = complex_input
            self.verbose = verbose
            self.builder = builder
            self.callback = callback

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
        :param max_func_evals: the maximum number of function evaluations.  If not specified will be 50 * len(initial).
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

        data = utils.Data(
            use_jacobian=jacobian == 'native',
            complex_input=complex_input,
            verbose=verbose,
            builder=builder,
            callback=callback,
        )

        max_func_evals = max_func_evals if max_func_evals is not None else 100 * len(initial)

        # Do it!
        res = optimize.root(
            functools.partial(self._jac2, func=fun, opt_data=data),
            x0_,
            jac=True,
            method='lm',
            options=dict(col_deriv=False, xtol=x_tol, maxiter=max_func_evals),
            # bounds=bounds,
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
            rmsd=0.0,
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
    def _jac2(state: functions.StateLike, func: functions.Function,
              opt_data: 'utils.Data') -> Tuple[np.ndarray, np.ndarray]:
        if opt_data.complex_input:
            state = utils.join_state(state)

        if opt_data.use_jacobian:
            value, jac = func(state, jacobian=True)
        else:
            value = func(state, jacobian=False)
            jac = None

        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))

        if opt_data.callback is not None:
            if opt_data.builder is None:
                inp = state
            else:
                inp = opt_data.builder(state)
            opt_data.callback(inp, value, jac)

        # Prepare for passing back to scipy
        value = utils.split_state(value)
        if jac is not None:
            jac = utils.split_jacobian(
                jac, complex_inputs=opt_data.complex_input, complex_outputs=np.iscomplexobj(value)
            )

        return value, jac
