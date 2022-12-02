# -*- coding: utf-8 -*-
import numpy as np
import sympy

from milad import functions


def test_function(
    function: functions.Function, value, expected_output=None, check_jacobian=None
):
    if function.input_type:
        assert issubclass(type(value), function.input_type)
    output = function(value)
    if function.output_type:
        assert issubclass(type(output), function.output_type)

    if expected_output is not None:
        assert np.all(functions.get_bare(output) == functions.get_bare(expected_output))

    if check_jacobian is None:
        check_jacobian = function.supports_jacobian

    # Try checking the inverse
    if function.inverse is not None:
        inverted = function.inverse(output)
        assert np.allclose(functions.get_bare(value), functions.get_bare(inverted))

    if check_jacobian:
        # Check the derivatives
        vector_input = functions.get_bare(value)
        reals = np.isreal(vector_input)
        # Create the symbolic inputs to the function
        input_vec = np.array(
            [sympy.Symbol(f"x{i}", real=real) for i, real in enumerate(reals)]
        )

        # Overwrite the input with a symbolic version
        if isinstance(value, np.ndarray):
            value = input_vec
        else:
            value.vector = input_vec

        output, jac = function(value, jacobian=True)
        output = functions.get_bare(output)
        if not isinstance(output, np.ndarray):
            # Must be a scalar
            output = [output]
        for i, out in enumerate(output):
            for j, variable in enumerate(input_vec):
                if np.ma.is_masked(jac[i, j]):
                    continue
                deriv = sympy.diff(out, variable)
                assert sympy_equal(jac[i, j], deriv)


def sympy_equal(expr1, expr2):
    if isinstance(expr1, sympy.Expr):
        expr1 = expr1.expand()
    expr2 = expr2.expand()
    difference = expr1 - expr2
    if not expr1 == expr2:
        if isinstance(difference, sympy.Number):
            return np.isclose(complex(difference), 0.0)

        # If they differ, check that it's by a meaningful amount.
        # We cast to complex here, even for reals, because this will catch both cases without having to do checks
        coeffs = np.array(
            tuple(difference.as_coefficients_dict().values()), dtype=np.complex
        )
        return np.allclose(coeffs, 0.0)

    return True


def generate_vectors_on_sphere(num):
    """Generate `num` random unit vectors on a sphere"""
    vecs = np.empty((num, 3))
    for i in range(num):
        vec = np.random.rand(3)
        vec /= np.linalg.norm(vec, axis=0)
        vecs[i] = vec
    return vecs
