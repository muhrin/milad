# -*- coding: utf-8 -*-
import shutil

import ase
import ase.build
import numpy as np
import scipy.special
import pytest

from milad import asetools
from milad import atomic
from milad import descriptors
from milad import generate
from milad import functions
from milad import fingerprinting
from milad import invariants
from milad import utils
from milad import zernike

MILAD_TO_AMP = 52.63789013914202
# Multiply by this strange factor that comes about because of minor differences between the way that we perform
# the calculations
MILAD_TO_AMP_RADIAL = 1.0 / np.sqrt(3.0)


def test_compare_zernike_with_amp():
    """Make sure we give the same results for the Zernike descriptor as AMP"""
    # pylint: disable=too-many-locals

    pytest.importorskip("amp")
    import amp.descriptor.zernike

    label = "amp-fp"
    milad_label = "amp-milad-fp"
    # Remove any files from the last test
    shutil.rmtree(f"./{label}-fingerprints.ampdb", ignore_errors=True)
    shutil.rmtree(f"./{label}-neighborlists.ampdb", ignore_errors=True)
    shutil.rmtree(f"./{milad_label}-fingerprints.ampdb", ignore_errors=True)
    shutil.rmtree(f"./{milad_label}-neighborlists.ampdb", ignore_errors=True)

    # Settings
    nmax = 7
    natoms = 6
    rcut = 2.8
    species = ("Si",)
    specie_numbers = (14,)
    specie_weight = 14.0
    pts = generate.random_points_in_sphere(natoms, radius=1.5)
    atoms = atomic.AtomsCollection(natoms, positions=pts, numbers=specie_numbers[0])

    # Create the descriptors
    amp_descriptor = amp.descriptor.zernike.Zernike(
        cutoff=rcut,
        Gs=dict(Si=dict(Si=specie_weight)),
        nmax=nmax,
        elements=species,
        fortran=False,
        dblabel=label,
    )

    # Create the MILAD descriptor
    milad_descriptor = descriptors.AmpDescriptor(
        fingerprinting.descriptor(
            features=dict(
                type=functions.WeightedDelta, kwargs=dict(weight=specie_weight)
            ),
            moments_calculator=zernike.ZernikeMomentsCalculator(7, use_direct=True),
            invs=invariants.PowerSpectrum(mix_radials=False, radials_first=True),
            cutoff=rcut,
            apply_cutoff=True,
            smooth_cutoff=True,
        ),
        dblabel="amp-milad-fp",
    )

    # Use AMP to calculate fingerprints
    system_dict = {"0": asetools.milad2ase(atoms)}
    amp_descriptor.calculate_fingerprints(
        system_dict, parallel=dict(cores=1), calculate_derivatives=False
    )

    milad_descriptor.calculate_fingerprints(
        system_dict, parallel=dict(cores=1), calculate_derivatives=False
    )

    for amp_fp, milad_fp in zip(
        amp_descriptor.fingerprints["0"], milad_descriptor.fingerprints["0"]
    ):
        assert amp_fp[0] == milad_fp[0]
        assert np.allclose(np.array(amp_fp[1]), MILAD_TO_AMP * np.array(milad_fp[1]))


def test_derivatives(complex_invariants):
    pytest.importorskip("amp")
    import amp.descriptor.zernike

    label = "amp-fp"
    # Remove any files from the last test
    shutil.rmtree(f"./{label}-fingerprints.ampdb", ignore_errors=True)
    shutil.rmtree(f"./{label}-fingerprint-primes.ampdb", ignore_errors=True)
    shutil.rmtree(f"./{label}-neighborlists.ampdb", ignore_errors=True)

    # Settings
    nmax = 7
    rcut = 3.0
    species = ("Si",)
    specie_weight = 14.0

    # Create the descriptors
    amp_descriptor = amp.descriptor.zernike.Zernike(
        cutoff=rcut,
        Gs=dict(Si=dict(Si=specie_weight)),
        nmax=nmax,
        elements=species,
        fortran=False,
        dblabel=label,
    )

    # Create the MILAD descriptor
    milad_descriptor = descriptors.AmpDescriptor(
        fingerprinting.descriptor(
            features=dict(
                type=functions.WeightedDelta, kwargs=dict(weight=specie_weight)
            ),
            moments_calculator=zernike.ZernikeMomentsCalculator(7),
            invs=complex_invariants,
            cutoff=rcut,
            apply_cutoff=True,
            smooth_cutoff=True,
        ),
        dblabel="amp-milad-fp",
    )

    systems = {"0": ase.build.bulk("Si", "hcp", a=2.6)}
    amp_descriptor.calculate_fingerprints(
        systems, parallel=dict(cores=1), calculate_derivatives=True
    )
    milad_descriptor.calculate_fingerprints(
        systems, parallel=dict(cores=1), calculate_derivatives=True
    )

    # Make sure we include the same interaction terms
    assert (
        set(amp_descriptor.fingerprintprimes["0"])
        - set(milad_descriptor.fingerprintprimes["0"])
    ) == set()


def test_amp_zernike_comparison():
    pytest.importorskip("amp")
    import amp.descriptor.zernike

    label = "amp-fp"
    # Remove any files from the last test
    shutil.rmtree(f"./{label}-fingerprints.ampdb", ignore_errors=True)
    shutil.rmtree(f"./{label}-neighborlists.ampdb", ignore_errors=True)

    nmax = 7
    natoms = 5
    rcut = 1.0

    pts = generate.random_points_in_sphere(natoms - 1)
    atoms = atomic.AtomsCollection(natoms - 1, positions=pts, numbers=14)

    milad_descriptor = functions.Chain(
        atomic.FeatureMapper(),
        functions.CosineCutoff(rcut),
        zernike.ZernikeMomentsCalculator(nmax, use_direct=True),
        invariants.PowerSpectrum(mix_radials=False, radials_first=True),
    )

    amp_descriptor = amp.descriptor.zernike.Zernike(
        cutoff=rcut,
        Gs=dict(Si=dict(Si=1.0)),
        nmax=nmax,
        elements=["Si"],
        fortran=False,
        dblabel=label,
    )

    system_dict = {
        "0": ase.Atoms(
            positions=np.concatenate((np.array([[0.0, 0.0, 0.0]]), pts)),
            numbers=[14] * natoms,
        )
    }
    amp_descriptor.calculate_fingerprints(
        system_dict, parallel=dict(cores=1), calculate_derivatives=False
    )

    phi = milad_descriptor(atoms)

    assert np.allclose(
        np.array(amp_descriptor.fingerprints["0"][0][1]), phi * MILAD_TO_AMP
    )


def test_radial_function():
    """Check that our Zernike radial function agrees with AMP"""
    # pylint: disable=invalid-name
    pytest.importorskip("amp")
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
            assert np.isclose(our_radial, MILAD_TO_AMP_RADIAL * amp_radial)
