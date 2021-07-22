# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np
import sympy

import milad
from milad import zernike
from milad import functions
from milad import generate
from milad import geometric


def test_invariant_single_mass(moment_invariants, request, save_figures):
    num_invariants = 20
    num_masses = 20

    origin = np.zeros((1, 3))

    fig, axes = plt.subplots()

    for i in range(num_masses):
        mass = 0. + (0.1 * i)

        invariants = milad.invariants.calc_moment_invariants(moment_invariants[:num_invariants], origin, 0.4, mass)

        milad.plot.plot_invariants(invariants, axes, label='mass={}'.format(mass))

    fig.legend()
    if save_figures:
        fig.savefig('{}.pdf'.format(request.node.name))


def test_invariant_two_weights(moment_invariants, request, save_figures):
    num_invariants = 64
    num_weights = 11

    positions = np.array(((-2., 0., 0.), (2., 0., 0.)))

    fig, axes = plt.subplots()

    for i in range(num_weights):
        mass = 0. + (0.1 * i)

        invariants = milad.invariants.calc_moment_invariants(
            moment_invariants[:num_invariants], positions, 0.4, (1., mass), normalise=True
        )

        milad.plot.plot_invariants(invariants, axes, label=f'$w={mass:.1f}$')

    axes.set_yscale('log')
    axes.set_title('Varying mass')
    fig.legend()
    if save_figures:
        fig.savefig('{}.pdf'.format(request.node.name))


def test_invariant_derivative(moment_invariants):
    # Take a random invariant and make sure that derivatives are getting calculated accurately
    invariant = random.choice(moment_invariants)

    # Symbols for moments
    m = sympy.IndexedBase('m')  # pylint: disable=invalid-name
    phi = invariant.apply(m)  # Analytic expression for moments
    derivatives = invariant.get_gradient()

    for indices, entry in derivatives.items():
        dm = m[indices]  # pylint: disable=invalid-name
        dphi_dm_analytic = phi.diff(dm)
        dphi_dm_calculated = entry(m)

        assert dphi_dm_calculated == dphi_dm_analytic


def test_invariants_function(moment_invariants):
    num_points = 10
    max_order = 10

    invariants_fn = moment_invariants  # milad.invariants.MomentInvariants(*moment_invariants)
    moments_fn = geometric.GeometricMomentsCalculator(max_order)

    pts = generate.random_points_in_sphere(num_points)
    env = functions.Features(*map(functions.WeightedDelta, pts))
    moments = moments_fn(env)
    phi = invariants_fn(moments)

    # The 1st moment is always the total mass
    assert phi[0] == num_points

    # Now try the same thing using chain
    combined_fn = functions.Chain(moments_fn, invariants_fn)
    phi2, _jacobian = combined_fn(env, jacobian=True)

    assert np.all(phi == phi2)


def test_invariants_derivatives_correctness(complex_invariants):
    complex_invariants = complex_invariants[:10]
    # The moments
    m = sympy.IndexedBase('m', complex=True)  # pylint: disable=invalid-name
    # Fill with symbols
    zernike_moms = milad.ZernikeMoments.from_indexed(m, complex_invariants.max_order, dtype=object)

    phi, jac = complex_invariants(zernike_moms, jacobian=True)

    for i, inv in enumerate(phi):
        for j, (_, mom) in enumerate(zernike_moms.iter(redundant=True)):
            # Perform symbolic derivative
            diff = sympy.diff(inv, mom).expand()

            # Get derivative from the Jacobian
            from_jac = jac[i, j]
            if not isinstance(from_jac, (int, np.integer)):
                from_jac = from_jac.expand()

            # Compare
            difference = diff - from_jac
            if not diff == from_jac:
                # If they differ, check that it's by a meaningful amount
                coeffs = np.array(tuple(difference.as_coefficients_dict().values()))
                np.testing.assert_array_almost_equal(coeffs, 0., decimal=10)


def test_against_chiral_tetrahedra(complex_invariants, chiral_tetrahedra):
    minus, plus = chiral_tetrahedra

    minus_moms = zernike.from_deltas(complex_invariants.max_order, minus)
    plus_moms = zernike.from_deltas(complex_invariants.max_order, plus)

    minus_phi = complex_invariants(minus_moms)
    plus_phi = complex_invariants(plus_moms)

    assert not np.allclose(minus_phi, plus_phi)
