import numpy as np
import pytest
import scipy
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from second_quantization import bosons, hilbert_space


@pytest.mark.parametrize("sparse", [False, True])
def test_cross_mode_commutation_zero(sparse, rng, as_dense):
    """[a_i, a_j^†]=0 for i≠j using the implemented truncated ladders."""
    a0, a1 = sympy.symbols("a0:2", cls=BosonOp)
    ops = [a0, a1]
    trunc = [int(rng.integers(3, 8)), int(rng.integers(3, 8))]

    expr_left = a0 * Dagger(a1)
    expr_right = Dagger(a1) * a0

    left = hilbert_space.to_matrix(expr_left, ops, sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]
    right = hilbert_space.to_matrix(expr_right, ops, sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]

    left_d = as_dense(left)
    right_d = as_dense(right)

    np.testing.assert_allclose(left_d - right_d, np.zeros_like(left_d))


@pytest.mark.parametrize("sparse", [False, True])
def test_single_mode_commutator_truncated(sparse, rng, as_dense):
    """Validate [a, a^†] with the exact ladder used in boson_basis (truncation-aware)."""
    a = BosonOp("a")
    trunc = int(rng.integers(3, 8))

    basis = bosons.boson_basis([a], truncation=trunc, sparse=sparse)
    A = basis[a]
    Ad = A.conj().T

    comm = A @ Ad - Ad @ A
    expected = hilbert_space.to_matrix(
        a * Dagger(a) - Dagger(a) * a, [a], sparse=sparse, truncation=trunc
    )[sympy.S.One]

    np.testing.assert_allclose(as_dense(comm), as_dense(expected))


@pytest.mark.parametrize("sparse", [False, True])
def test_power_commutator_truncated(sparse, rng, as_dense):
    """Check [a^2, a^†] using the same ladder matrices to ensure internal consistency."""
    a = BosonOp("a")
    trunc = int(rng.integers(3, 10))

    basis = bosons.boson_basis([a], truncation=trunc, sparse=sparse)
    A2 = basis[a**2]
    Ad = basis[a].conj().T

    comm = A2 @ Ad - Ad @ A2
    expected = hilbert_space.to_matrix(
        a**2 * Dagger(a) - Dagger(a) * a**2, [a], sparse=sparse, truncation=trunc
    )[sympy.S.One]

    np.testing.assert_allclose(as_dense(comm), as_dense(expected))


@pytest.mark.parametrize("sparse", [False, True])
def test_parity_dimensions_and_values(sparse, rng, as_dense):
    """Parity matrix equals N modulo two on the truncated tensor basis."""
    ops = [BosonOp("a0"), BosonOp("a1")]
    trunc = [int(rng.integers(3, 8)), int(rng.integers(3, 8))]
    parity = bosons.boson_parity(ops, truncation=trunc, sparse=sparse)

    parity_d = as_dense(parity)

    dims = np.prod([t + 1 for t in trunc])
    assert parity_d.shape == (dims, dims)

    diag = np.diag(parity_d)
    occupations = []
    for idx in range(dims):
        tmp = idx
        total = 0
        for t in trunc[::-1]:
            total += tmp % (t + 1)
            tmp //= t + 1
        occupations.append(total)
    expected_diag = np.array(occupations) % 2
    np.testing.assert_allclose(diag, expected_diag)


@pytest.mark.parametrize("sparse", [False, True])
def test_parity_unequal_truncations(sparse, as_dense):
    """Parity follows the tensor-product ordering for unequal truncations."""
    ops = [BosonOp("a0"), BosonOp("a1")]
    trunc = [1, 2]

    parity = bosons.boson_parity(ops, truncation=trunc, sparse=sparse)
    expected = np.diag([0, 1, 0, 1, 0, 1])

    np.testing.assert_array_equal(as_dense(parity), expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_parity_scalar_truncation_and_mismatch_error(sparse, as_dense):
    """Scalar truncation should broadcast; mismatched list should raise."""
    ops = [BosonOp("a0"), BosonOp("a1")]

    parity = bosons.boson_parity(ops, truncation=3, sparse=sparse)
    dims = (3 + 1) ** 2
    assert as_dense(parity).shape == (dims, dims)

    with pytest.raises(ValueError):
        bosons.boson_parity(ops, truncation=[3], sparse=sparse)


@pytest.mark.parametrize("sparse", [False, True])
def test_invalid_truncation_length_raises(sparse):
    """Mismatch between truncation list and operator list should raise ValueError."""
    ops = [BosonOp("a0"), BosonOp("a1")]
    with pytest.raises(ValueError):
        bosons.boson_basis(ops, truncation=[2], sparse=sparse)


@pytest.mark.parametrize("sparse", [False, True])
def test_boson_basis_sparse_power_pattern(sparse, as_dense):
    """a^2 ladder has expected sqrt(n(n-1)) factors on superdiagonal."""
    a = BosonOp("a")
    trunc = 5

    basis = bosons.boson_basis([a], truncation=trunc, sparse=sparse)
    a2 = basis[a**2]
    dense = as_dense(a2)

    # Expect superdiagonal at k=2 with entries sqrt(n(n-1))
    expected = np.zeros_like(dense)
    n_max = trunc
    for n in range(2, n_max + 1):
        expected[n - 2, n] = np.sqrt(n * (n - 1))

    np.testing.assert_allclose(dense, expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("truncation", [0, 2])
def test_boson_powers_above_cutoff_vanish(sparse, truncation, as_dense):
    a = BosonOp("a")
    for operator in (a, Dagger(a)):
        matrix = hilbert_space.to_matrix(
            operator ** (truncation + 1), [a], sparse=sparse, truncation=truncation
        )
        assert set(matrix) == {sympy.S.One}
        np.testing.assert_array_equal(
            as_dense(matrix[1]), np.zeros((truncation + 1,) * 2)
        )


@pytest.mark.parametrize("sparse", [False, True])
def test_boson_decomposition_coefficients(sparse, rng):
    """Matrix -> boson inversion should recover normal-ordered coefficients."""
    a = BosonOp("a")
    trunc = int(rng.integers(4, 7))
    tol = 1e-8

    expr = (
        2.0
        + 1.3 * Dagger(a)
        + 0.7 * a
        + 3.0 * Dagger(a) * a
        + 0.5 * Dagger(a) ** 2 * a**2
    )

    matrix = hilbert_space.to_matrix(expr, [a], sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]
    decomposition = bosons.BosonDecomposition(matrix, truncation=trunc)

    expected_keys = {((0, 0),), ((1, 0),), ((0, 1),), ((1, 1),), ((2, 2),)}
    for key in expected_keys:
        assert key in decomposition
    assert decomposition[((0, 0),)] == pytest.approx(2.0, abs=tol)
    assert decomposition[((1, 0),)] == pytest.approx(1.3, abs=tol)
    assert decomposition[((0, 1),)] == pytest.approx(0.7, abs=tol)
    assert decomposition[((1, 1),)] == pytest.approx(3.0, abs=tol)
    assert decomposition[((2, 2),)] == pytest.approx(0.5, abs=tol)

    unexpected = {k: v for k, v in decomposition.items() if k not in expected_keys}
    for key, value in unexpected.items():
        assert abs(value) < tol


@pytest.mark.parametrize("sparse", [False, True])
def test_boson_to_operators_numeric_round_trip(sparse, as_dense):
    a = BosonOp("a")
    matrix = np.array([[0, 1.1289722927208916 - 1.1332872140034806j], [0, 0]])
    if sparse:
        matrix = scipy.sparse.csr_array(matrix)
    expression = bosons.boson_to_operators(matrix, [a], truncation=1)
    recovered = hilbert_space.to_matrix(expression, [a], sparse=sparse, truncation=1)
    actual = sum(complex(key) * as_dense(value) for key, value in recovered.items())
    np.testing.assert_allclose(actual, as_dense(matrix), rtol=1e-12, atol=1e-12)
