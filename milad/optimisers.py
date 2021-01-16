# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize

from . import functions

__all__ = ('LeastSquaresOptimiser',)


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

    def optimise(self, func: functions.Function, initial: functions.StateLike, jacobian='2-point', verbose=False):
        """
        :param func: the function to optimise
        :param initial: the initial state
        :param jacobian: if 'native' the analytic Jacobian will be requested from 'func', otherwise this option is
            passed to optimize.least_squares
        :return:
        """

        # SciPy can only deal with numpy arrays if we get one, all good, otherwise
        # we have to rely on a builder
        if isinstance(initial, np.ndarray):
            fun = func
            x0_ = initial
        else:
            # Need to add a builder step
            builder = initial.builder
            fun = functions.Chain(builder, func)
            x0_ = builder.inverse(initial)

        # Annoyingly scipy least_squares calls the function and the separately
        # but we don't support getting the Jacobian on it's own (it always comes
        # with a function evaluation and we don't want to call it twice so just
        # cache the result)
        jac = self._jac if jacobian == 'native' else '2-point'
        data = LeastSquaresOptimiser.Data(use_jacobian=jacobian == 'native', verbose=verbose)

        # Do it!
        return optimize.least_squares(self._calc, x0_, jac=jac, kwargs=dict(func=fun, opt_data=data))

    def optimise_target(
        self,
        func: functions.Function,
        initial: functions.StateLike,
        target: functions.StateLike,
        jacobian='2-point',
        verbose=False
    ):
        """
        Optimise the function to a given target value.  This minimises the loss of the difference between the function
        and the target.

        :param func: the function to optimise
        :param initial: the initial state
        :param target: the output state
        :param jacobian: if 'native' the analytic Jacobian will be requested from 'func', otherwise this option is
            passed to optimize.least_squares
        :return:
        """
        # Calculate residuals to a particular target
        return self.optimise(
            functions.Chain(func, functions.Residuals(target)), initial=initial, jacobian=jacobian, verbose=verbose
        )

    def _calc(
        self, state: functions.StateLike, func: functions.Function, opt_data: 'LeastSquaresOptimiser.Data'
    ) -> np.ndarray:
        # pylint: disable=no-self-use
        res = func(state, jacobian=opt_data.use_jacobian)
        if opt_data.use_jacobian:
            value, opt_data.last_jacobian = res
        else:
            value = res

        opt_data.last_state = state
        value = value.real
        if opt_data.verbose:
            print('|Max| {}'.format(np.abs(value).max()))
        return value

    def _jac(
        self, state: functions.StateLike, func: functions.Function, opt_data: 'LeastSquaresOptimiser.Data'
    ) -> np.ndarray:
        if np.all(opt_data.last_jacobian[0] == state):
            return opt_data.last_jacobian[1]

        # Reuse calc to do the function call and updating of the cache
        self._calc(state, func, opt_data)
        return opt_data.last_jacobian
