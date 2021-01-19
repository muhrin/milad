# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np
import sympy

import milad
from milad import functions
from milad import generate
from milad import geometric


def test_invariant_single_mass(moment_invariants, request):
    num_invariants = 20
    num_masses = 20

    origin = np.zeros((1, 3))

    fig, axes = plt.subplots()

    for i in range(num_masses):
        mass = 0. + (0.1 * i)

        invariants = milad.invariants.calc_moment_invariants(moment_invariants[:num_invariants], origin, 0.4, mass)

        milad.plot.plot_invariants(invariants, axes, label='mass={}'.format(mass))

    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_invariant_two_weights(moment_invariants, request):
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
    fig.savefig('{}.pdf'.format(request.node.name))


def test_invariant_derivative(moment_invariants):
    # Take a random invariant and make sure that derivatives are getting calculated accurately
    invariant = random.choice(moment_invariants)

    m = sympy.IndexedBase('m')  # Symbols for moments
    phi = invariant.apply(m)  # Analytic expression for moments
    derivatives = invariant.derivatives()

    for indices, entry in derivatives.items():
        dm = m[indices]
        dphi_dm_analytic = phi.diff(dm)
        dphi_dm_calculated = entry.apply(m)

        assert dphi_dm_calculated == dphi_dm_analytic


def test_invariants_function(moment_invariants):
    NUM_POINTS = 10
    MAX_ORDER = 10

    invariants_fn = milad.invariants.MomentInvariants(*moment_invariants)
    moments_fn = geometric.GeometricMomentsCalculator(MAX_ORDER)

    pts = generate.random_points_in_sphere(NUM_POINTS)
    env = functions.Features(*map(functions.WeightedDelta, pts))
    moments = moments_fn(env)
    phi = invariants_fn(moments)

    # The 1st moment is always the total mass
    assert phi[0] == NUM_POINTS

    # Now try the same thing using chain
    combined_fn = functions.Chain(moments_fn, invariants_fn)
    phi2, _jacobian = combined_fn(env, jacobian=True)

    assert np.all(phi == phi2)
