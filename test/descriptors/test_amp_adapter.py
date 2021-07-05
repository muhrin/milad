# -*- coding: utf-8 -*-
import numpy as np
import pytest

from milad import asetools
from milad import atomic
from milad import descriptors
from milad import generate
from milad import functions
from milad import fingerprinting
from milad import invariants_
from milad import zernike


def test_compare_zernike_with_amp():
    """Make sure we give the same results for the Zernike descriptor as AMP"""
    pytest.importorskip('amp')
    import amp.descriptor.zernike

    # Settings
    nmax = 7
    natoms = 6
    rcut = 2.8
    species = ('Si',)
    specie_numbers = (14,)
    specie_weight = 14.
    pts = generate.random_points_in_sphere(natoms, radius=1.5)
    atoms = atomic.AtomsCollection(natoms, positions=pts, numbers=specie_numbers[0])

    # Create the descriptors
    amp_descriptor = amp.descriptor.zernike.Zernike(
        cutoff=rcut, Gs=dict(Si=dict(Si=specie_weight)), nmax=nmax, elements=species, fortran=False, dblabel='amp-fp'
    )

    # Create the MILAD descriptor
    milad_descriptor = descriptors.AmpDescriptor(
        fingerprinting.descriptor(
            features=dict(type=functions.WeightedDelta, kwargs=dict(weight=specie_weight)),
            moments_calculator=zernike.ZernikeMomentCalculator(7, use_direct=True),
            invs=invariants_.PowerSpectrum(mix_radials=False, radials_first=True),
            cutoff=rcut,
            apply_cutoff=True,
            smooth_cutoff=True
        ),
        dblabel='amp-milad-fp'
    )

    # Use AMP to calcaulte fingerprints
    system_dict = {'0': asetools.milad2ase(atoms)}
    amp_descriptor.calculate_fingerprints(system_dict, parallel=dict(cores=1), calculate_derivatives=False)

    milad_descriptor.calculate_fingerprints(system_dict, parallel=dict(cores=1), calculate_derivatives=False)

    for amp_fp, milad_fp in zip(amp_descriptor.fingerprints['0'], milad_descriptor.fingerprints['0']):
        assert amp_fp[0] == milad_fp[0]
        assert np.allclose(np.array(amp_fp[1]), np.array(milad_fp[1]))
