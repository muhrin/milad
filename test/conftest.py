import pytest

import milad


@pytest.fixture(scope="session")
def moment_invariants():
    invariants = milad.invariants.read_invariants()
    yield invariants
