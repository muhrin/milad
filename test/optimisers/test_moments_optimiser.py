# -*- coding: utf-8 -*-
from milad import generate
from milad import optimisers
from milad import zernike


def test_moments_optimiser(complex_invariants):
    """Basic tests for the moments optimiser"""
    pts = generate.random_points_in_sphere(10, 1.)

    # Create some Zernike moments from delta functions
    moms = zernike.from_deltas(complex_invariants.max_order, pts)
    # Create the invariants vector
    target = complex_invariants(moms)

    # Make settings very loose so this is fast, we're just testing the logic here...
    optimiser = optimisers.MomentsOptimiser()
    result = optimiser.optimise(
        invariants_fn=complex_invariants,
        target=target,
        initial=zernike.ZernikeMoments.rand(complex_invariants.max_order),
        target_rmsd=1e-2,
        cost_tol=1e-2,
        grad_tol=1e-2,
        jacobian='2-point',
        verbose=True,
    )
    assert result.success, result.message
