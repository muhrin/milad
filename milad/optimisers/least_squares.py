# -*- coding: utf-8 -*-
import collections
import math
from typing import Tuple, Optional

import numpy as np
from scipy import optimize

from milad import functions

__all__ = ('LeastSquaresOptimiser',)

OptimiserResult = collections.namedtuple('OptimiserResult', 'success message value rmsd n_func_eval n_jac_eval')

BoundsType = Tuple[Optional[functions.StateLike], Optional[functions.StateLike]]


class LeastSquaresOptimiser:
    """Uses scipy optimize.least_squares to perform optimistaion"""

    class Data:
        """Data used during an optimisation"""

        def __init__(self, use_jacobian, complex_input: bool, verbose=False):
            self.use_jacobian = use_jacobian
            self.complex_input = complex_input
            self.verbose = verbose

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
            x0_ = split_state(x0_)

        # Annoyingly scipy least_squares calls the function and the separately
        # but we don't support getting the Jacobian on it's own (it always comes
        # with a function evaluation and we don't want to call it twice so just
        # cache the result)
        jac = self._jac if jacobian == 'native' else jacobian
        data = LeastSquaresOptimiser.Data(
            use_jacobian=jacobian == 'native', complex_input=complex_input, verbose=verbose
        )

        max_func_evals = max_func_evals if max_func_evals is not None else 100 * len(initial)

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
            value = join_state(res.x)
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
            verbose=verbose,
        )

    @staticmethod
    def _calc(
        state: functions.StateLike, func: functions.Function, opt_data: 'LeastSquaresOptimiser.Data'
    ) -> np.ndarray:
        if opt_data.complex_input:
            state = join_state(state)

        value = functions.get_bare(func(state))
        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))

        return split_state(value)

    @staticmethod
    def _jac(
        state: functions.StateLike, func: functions.Function, opt_data: 'LeastSquaresOptimiser.Data'
    ) -> np.ndarray:
        if opt_data.complex_input:
            state = join_state(state)

        value, jac = func(state, jacobian=True)
        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))

        split = split_jacobian(jac, complex_inputs=opt_data.complex_input, complex_outputs=np.iscomplexobj(value))

        return split


def join_state(state: np.ndarray) -> np.ndarray:
    half_size = int(len(state) / 2)
    out_state = state[:half_size] + 1j * state[half_size:]
    return out_state


def split_state(state: np.ndarray) -> np.ndarray:
    """Split is a state vector that is (potentially) complex.  If it is the vector is double in rows to be
    (reals, imags), otherwise the vector is simply returned unaltered."""
    if np.iscomplexobj(state):
        if isinstance(state, np.ndarray):
            return np.concatenate((state.real, state.imag))
        return np.array((state.real, state.imag))

    return state


def split_jacobian(jacobian: np.ndarray, complex_inputs: bool, complex_outputs: bool) -> np.ndarray:
    """Assuming that the function is analytic split the Jacobian into parts corresponding to a combination of
    complex inputs/outputs.  The layout of the return Jacobian is:

    | d Re(f)   d Re(f) |
    | -------   ------- |
    | d Re(x)   d Im(x) |
    |                   |
    | d Im(f)   d Im(f) |
    | -------   ------- |
    | d Re(x)   d Im(x) |

    If the function has complex inputs and real outputs then only the first row is return.
    If the function has real inputs and complex outputs then only the first column is return.
    Otherwise the full Jacobian is returned.

    See the Cauchy-Riemann equations for more details:
    https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations
    """
    if not complex_inputs and not complex_outputs:
        return jacobian.real

    orig_size = jacobian.shape
    if len(orig_size) == 1:
        orig_size = (orig_size[0], 1)
    new_size = list(orig_size)
    if complex_outputs:
        new_size[0] *= 2
    if complex_inputs:
        new_size[1] *= 2

    jac = np.empty(new_size)
    # Input: Real, Output: Real
    jac[:orig_size[0], :orig_size[1]] = jacobian.real
    if complex_outputs:
        # Input: Real, Output: Imag
        jac[orig_size[0]:, :orig_size[1]] = jacobian.imag
    if complex_inputs:
        complex_part = jacobian * 1j
        # Input: Imag, Output: Real
        jac[:orig_size[0], orig_size[1]:] = complex_part.real
        if complex_outputs:
            # Input: Imag, Output: Imag
            jac[orig_size[0]:, orig_size[1]:] = complex_part.imag

    return jac