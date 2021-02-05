# -*- coding: utf-8 -*-
import numpy as np

import ase.build
import ase.visualize

import milad
from milad import functions
from milad import zernike
from milad.play import asetools


def test_structure_optimiser_invariants(complex_invariants):
    """Basic tests of the structure optimiser that is given a set of invariants"""
    radius_factor = 1.25
    molecule = ase.build.molecule('C3H9N')
    max_radius = asetools.prepare_molecule(molecule)
    milad_molecule = asetools.ase2milad(molecule)
    cutoff = radius_factor * max_radius

    descriptor = milad.descriptor(
        species={'map': {
            'numbers': tuple(molecule.numbers),
            'range': (1., 5.)
        }},
        features={
            'type': functions.WeightedDelta,
            'map_species_to': 'WEIGHT'
        },
        cutoff=cutoff,
        invs=complex_invariants,
        moments_calculator=zernike.ZernikeMomentCalculator(complex_invariants.max_order),
        apply_cutoff=False,
    )
    fingerprint = descriptor(milad_molecule)

    optimiser = milad.optimisers.StructureOptimiser()
    initial = milad_molecule.copy()
    initial.positions += 0.1 * (np.random.rand(*initial.positions.shape) - 0.5)
    result = optimiser.optimise(
        descriptor=descriptor,
        target=fingerprint,
        initial=initial,
        x_tol=1e-5,
        verbose=True,
    )
    print(result.message)
    assert result.rmsd < 1e-3


def test_structure_optimiser_moments(complex_invariants):
    """Basic tests of the structure optimiser that is given a set of moments"""
    radius_factor = 1.25
    molecule = ase.build.molecule('C3H9N')
    max_radius = asetools.prepare_molecule(molecule)
    milad_molecule = asetools.ase2milad(molecule)
    cutoff = radius_factor * max_radius

    descriptor = milad.descriptor(
        species={'map': {
            'numbers': tuple(molecule.numbers),
            'range': (0.5, 5.)
        }},
        features={
            'type': functions.WeightedDelta,
            'map_species_to': 'WEIGHT'
        },
        cutoff=cutoff,
        invs=complex_invariants,
        moments_calculator=zernike.ZernikeMomentCalculator(complex_invariants.max_order),
        apply_cutoff=False,
    )
    target = descriptor.get_moments(milad_molecule)

    optimiser = milad.optimisers.StructureOptimiser()
    initial = milad_molecule.copy()
    initial.positions += 0.1 * (np.random.rand(*initial.positions.shape) - 0.5)

    # Let's test the mask by removing the species degrees of freedom
    mask = initial.get_mask()
    mask.numbers = initial.numbers

    result = optimiser.optimise(
        descriptor=descriptor,
        target=target,
        initial=initial,
        x_tol=1e-4,
        mask=mask,
        verbose=False,
    )
    print(result.message)
    assert result.rmsd < 1e-3
