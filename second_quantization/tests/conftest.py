import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    """Shared deterministic RNG for truncation choices in tests."""
    return np.random.default_rng(0)


@pytest.fixture()
def as_dense():
    """Return a helper that converts sparse matrices to dense arrays for assertions."""

    def _as_dense(mat):
        return mat.toarray() if hasattr(mat, "toarray") else mat

    return _as_dense
