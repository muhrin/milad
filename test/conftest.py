# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest

import milad
from milad import invariants, generate

READ_MAX = 128  # The maximum number of invariants to read from the file


@pytest.fixture(scope='session')
def moment_invariants():
    """Get geometric moment invariants"""
    invs = milad.invariants.read(filename=invariants.GEOMETRIC_INVARIANTS, read_max=READ_MAX)
    yield invs


@pytest.fixture(scope='session')
def complex_invariants():
    """Get complex moment invariants"""
    invs = milad.invariants.read(filename=invariants.COMPLEX_INVARIANTS)
    yield invs


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(1234)
    np.random.seed(1234)


@pytest.fixture()
def save_figures():
    """Return True if you want to save figures from tests"""
    return False


@pytest.fixture()
def chiral_tetrahedra():
    """Create chiral structures that cannot be distinguished by bispectrum from
    https://link.aps.org/doi/10.1103/PhysRevLett.125.166001
    """
    return generate.chiral_tetrahedra()
