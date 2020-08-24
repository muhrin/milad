# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy
from scipy.spatial.transform import Rotation

import milad


def generate_vectors_on_sphere(num):
    vecs = numpy.empty((num, 3))
    for i in range(num):
        vec = numpy.random.rand(3)
        vec /= numpy.linalg.norm(vec, axis=0)
        vecs[i] = vec
    return vecs


def test_env_simple_no_cutoff_fn(moment_invariants, request):
    # Settings
    numpy.random.seed(5)
    num_atoms = 6
    sigma = 0.5
    volume = 1.
    num_rotations = 10
    sphere_radius = 2.5
    # moment_invariants = moment_invariants[:20]
    normalise = True

    positions = sphere_radius * generate_vectors_on_sphere(num_atoms)

    # Populate the environment
    env = milad.play.SmoothGaussianEnvironment()
    for pos in positions:
        env.add_gaussian(pos, sigma=sigma, weight=volume)

    # The original
    series = [
        env.calc_moment_invariants(moment_invariants, normalise=normalise)
    ]

    for _ in range(num_rotations):
        # Generated rotated positions
        rot = Rotation.random()
        rotated = rot.apply(positions)

        # Create the environment
        rotated_env = milad.play.SmoothGaussianEnvironment()
        rotated_env.add_gaussians(rotated, sigma=sigma, mass=volume)

        # Calculate the invariants
        series.append(
            rotated_env.calc_moment_invariants(moment_invariants,
                                               normalise=normalise))

    fig, axes = plt.subplots()
    milad.plot.plot_multiple_invariants(series, axes)
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_env_simple_cos_cutoff_fn(moment_invariants, request):
    num_atoms = 6
    sigma = 1.0
    num_rotations = 10
    pos_sphere = 2.5
    # moment_invariants = moment_invariants[:20]
    cutoff_params = milad.play.make_cutoff_params(milad.play.cos_cutoff, 6.0)

    positions = pos_sphere * generate_vectors_on_sphere(num_atoms)

    # Populate the environment
    env = milad.play.SmoothGaussianEnvironment(**cutoff_params)
    for pos in positions:
        env.add_gaussian(pos, sigma=sigma)

    # The original
    series = [env.calc_moment_invariants(moment_invariants, normalise=True)]

    for _ in range(num_rotations):
        # Generated rotated positions
        rot = Rotation.random()
        rotated = rot.apply(positions)

        # Create the environment
        rotated_env = milad.play.SmoothGaussianEnvironment(**cutoff_params)
        results = rotated_env.add_gaussians(rotated, sigma)
        print('Results {}'.format(_))
        print(results)

        # Calculate the invariants
        series.append(
            rotated_env.calc_moment_invariants(moment_invariants,
                                               normalise=True))

    fig, axes = plt.subplots()
    milad.plot.plot_multiple_invariants(series, axes)
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_atom_entering(moment_invariants, request):
    """Test how the invariants change as an atom enters the cutoff"""
    num_atoms = 6
    cutoff = 6.
    sigma = 1.0

    start_x = 6.1
    delta = 0.05
    num_steps = 7

    weight = 200.
    normalise = True
    # moment_invariants = moment_invariants[:20]
    cutoff_params = milad.play.make_cutoff_params(milad.play.cos_cutoff,
                                                  cutoff)

    # The second atom just outside the cutoff zone
    positions = cutoff * numpy.random.rand(num_atoms, 3)
    target = num_atoms - 1
    positions[target] = (start_x, 0., 0.)

    # The original
    series = []

    for i in range(num_steps):
        x_pos = start_x - i * delta
        positions[target] = numpy.array((x_pos, 0., 0.))

        # Create the environment
        env = milad.play.SmoothGaussianEnvironment(**cutoff_params)
        # Add all the environment atoms
        for pos in positions[:target]:
            env.add_gaussian(pos, sigma=sigma, weight=weight)

        result = env.add_gaussian(positions[target],
                                  sigma=sigma,
                                  weight=weight)
        if x_pos > 6.:
            assert result is False

        # Calculate the invariants
        series.append(
            env.calc_moment_invariants(moment_invariants, normalise=normalise))

    diffs = [series[idx] - series[0] for idx in range(1, num_steps)]
    labels = [f'x={start_x - i * delta:.2f}' for i in range(1, num_steps)]

    fig, axes = plt.subplots()
    axes.set_title('Atom entering environment, $r_c=6$')
    # axes.set_yscale('log')
    axes.set_ylabel(f'Difference from x={start_x:.2f}')
    milad.plot.plot_multiple_invariants(diffs, axes, labels)
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_asymmetric_distribution(moment_invariants, request):
    """Test how the invariants change as an atom enters the cutoff"""
    sigma = 1.0
    normalise = True
    # moment_invariants = moment_invariants[:10]

    # The second atom just outside the cutoff zone
    pos = numpy.array((-3., 0., 0.))

    # Create the environment
    env = milad.play.SmoothGaussianEnvironment()
    env.add_gaussian(pos, sigma=sigma)

    series = []

    # Calculate the invariants
    series.append(
        env.calc_moment_invariants(moment_invariants, normalise=normalise))

    fig, axes = plt.subplots()
    milad.plot.plot_multiple_invariants(series, axes)
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_fingerprint(moment_invariants, request):
    num_atoms = 5
    num_rotations = 10
    scale = 4.
    num_invariants = len(moment_invariants)

    positions = scale * numpy.random.rand(num_atoms, 3)
    calculator = milad.play.FingerprintCalculator(invariants=moment_invariants,
                                                  sigma=1.,
                                                  cutoff=6.,
                                                  cutoff_function='cos',
                                                  normalise=True)

    orig_fp = calculator.calculate(positions)
    zeros = numpy.zeros((len(orig_fp)))

    series = [orig_fp]
    for _ in range(num_rotations):
        # Generated rotated positions
        rot = Rotation.random()
        rotated = rot.apply(positions)

        # Calculate the fingerprint
        fp = calculator.calculate(rotated)
        series.append(fp)

        delta = fp - orig_fp
        numpy.testing.assert_almost_equal(delta, zeros)

    fig, axes = plt.subplots()
    axes.set_title('5 random atoms 6x6x6 cube, $\sigma=1, r_{cut} = 6.$')
    milad.plot.plot_multiple_invariants(
        series,
        axes,
        labels=['$R_{{{}}}$'.format(idx) for idx in range(len(series))])
    axes.set_xlabel('Total fingerprint ({} atoms x {} invariants)'.format(
        num_atoms, num_invariants))
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))
