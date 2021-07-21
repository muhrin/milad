# -*- coding: utf-8 -*-
import ase
import numpy as np
import scipy.special
import pytest

from milad import functions
from milad import generate
from milad import atomic
from milad import invariants_
from milad import zernike
from milad import utils

# pylint: disable=invalid-name


def test_amp_zernike_comparison():
    pytest.importorskip('amp')
    import amp.descriptor.zernike

    nmax = 7
    natoms = 5
    rcut = 1.0

    pts = generate.random_points_in_sphere(natoms - 1)
    atoms = atomic.AtomsCollection(natoms - 1, positions=pts, numbers=14)

    milad_descriptor = functions.Chain(
        atomic.FeatureMapper(), functions.CosineCutoff(rcut), zernike.ZernikeMomentsCalculator(nmax, use_direct=True),
        invariants_.PowerSpectrum(mix_radials=False, radials_first=True)
    )

    amp_descriptor = amp.descriptor.zernike.Zernike(
        cutoff=rcut,
        Gs=dict(Si=dict(Si=1.)),
        nmax=nmax,
        elements=['Si'],
        fortran=False,
    )

    system_dict = {'0': ase.Atoms(positions=np.concatenate((np.array([[0., 0., 0.]]), pts)), numbers=[14] * natoms)}
    amp_descriptor.calculate_fingerprints(system_dict, parallel=dict(cores=1), calculate_derivatives=False)

    phi = milad_descriptor(atoms)

    assert np.allclose(np.array(amp_descriptor.fingerprints['0'][0][1]), phi)


def test_radial_function():
    """Check that our Zernike radial function agrees with AMP"""
    pytest.importorskip('amp')
    import amp.descriptor.zernike

    fac = scipy.special.factorial

    nmax = 7
    rho = 0.6353

    # Cache used by AMP
    factorial = [fac(0.5 * _) for _ in range(4 * nmax + 3)]

    for n in utils.inclusive(nmax):
        for l in utils.inclusive(n):
            amp_radial = amp.descriptor.zernike.calculate_R(n, l, rho, factorial)
            our_radial = zernike.r_nl(n, l, rho)
            assert np.isclose(our_radial, amp_radial)
