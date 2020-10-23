# -*- coding: utf-8 -*-
import ase.build
import ase.io
import matplotlib.pyplot as plt

import milad
from milad.play import asetools


def test_simple_atoms_fp(moment_invariants):
    cu_fcc = ase.build.bulk('Cu', 'sc', a=2.)
    ase.io.write('cu_sc.cif', cu_fcc)

    calculator = milad.play.FingerprintCalculator(moment_invariants, cutoff=2.5, cutoff_function=None)
    fps = asetools.calculate_fingerprint(cu_fcc, calculator)
    fps2 = asetools.calculate_fingerprints_dscribe(cu_fcc, calculator)

    diff = fps - fps2

    assert diff.max() == 0.
    assert len(fps) == 1


def test_multiple_species(moment_invariants, request):
    molecule = ase.build.molecule('CH3CH2OH')
    species = ['C', 'H', 'O']
    symbols = molecule.symbols
    num_atoms = len(symbols)

    fp = asetools.MiladFingerprint(species, moment_invariants, sigmas={'C': 1., 'H': 0.5, 'O': 0.7})

    fingerprints = fp.create(molecule)

    labels = []
    for i in range(num_atoms):
        labels.append('{}$_{}$'.format(i, symbols[i]))

    assert (fingerprints[3] - fp.create_single(molecule, 3)).max() == 0

    fig, axes = plt.subplots()
    axes.set_yscale('log')
    milad.plot.plot_multiple_invariants(fingerprints, axes, labels=labels)
    axes.set_title('C$_2$H$_6$O')
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_multiple_species_split(moment_invariants, request):
    molecule = ase.build.molecule('CH3CH2OH')
    species = ['C', 'H', 'O']

    fp = asetools.MiladFingerprint(
        species, moment_invariants, sigmas={
            'C': 0.4,
            'H': 0.3,
            'O': 0.5
        }, split_specie_pairs=True, cutoffs=100.
    )

    fingerprint = fp.create_single(molecule, 2)

    fig, axes = plt.subplots()
    axes.set_yscale('log')
    axes.set_title(str(molecule))
    milad.plot.plot_multiple_invariants(fingerprint, axes, labels=species)
    fig.legend()
    fig.savefig('{}.pdf'.format(request.node.name))


def test_generate_environments():
    molecule = ase.build.molecule('CH3CH2OH')

    envs = list(asetools.extract_environments(molecule, cutoff=4.))
    assert len(envs) == len(molecule)


def test_moments_calculator():
    molecule = ase.build.molecule('CH3CH2OH')

    # Get the environments
    envs = list(asetools.extract_environments(molecule, cutoff=4.))
    calculator = asetools.MomentsCalculator(milad.zernike.from_deltas, max_order=7)

    for env in envs:
        moments = calculator.calculate_moments(env)
