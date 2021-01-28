# -*- coding: utf-8 -*-
import pytest

import milad
from milad import invariants

READ_MAX = 128  # The maximum number of invariants to read from the file


@pytest.fixture(scope='session')
def moment_invariants():
    """Get geometric moment invariants"""
    invs = milad.invariants.read(filename=invariants.GEOMETRIC_INVARIANTS, read_max=READ_MAX)
    yield invs


@pytest.fixture(scope='session')
def complex_invariants():
    """Get complex moment invariants"""
    invs = milad.invariants.read(filename=invariants.COMPLEX_INVARIANTS, read_max=READ_MAX)
    yield invs
