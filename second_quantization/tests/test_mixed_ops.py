import numpy as np
import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.boson import BosonOp

from second_quantization import bosons, fermions, hilbert_space


@pytest.mark.parametrize("sparse", [False, True])
def test_boson_fermion_commutation(sparse, rng, as_dense):
    """Check [c_i, a_j] = 0 by comparing c a with a c when operators are ordered as [c, a]."""
    c = FermionOp("c")
    a = BosonOp("a")
    ops = [c, a]
    trunc = int(rng.integers(3, 8))

    expr_ca = c * a
    expr_ac = a * c

    mat_ca = hilbert_space.to_matrix(expr_ca, ops, sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]
    mat_ac = hilbert_space.to_matrix(expr_ac, ops, sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]

    np.testing.assert_allclose(as_dense(mat_ca), as_dense(mat_ac))


@pytest.mark.parametrize("sparse", [False, True])
def test_fermion_sign_with_boson_spectator(sparse, rng, as_dense):
    """Ensure fermionic sign survives in mixed systems: c_1^† c_0 a^† = - c_0 c_1^† a^† for operators [c_0, c_1, a]."""
    c0, c1 = sympy.symbols("c0:2", cls=FermionOp)
    a = BosonOp("a")
    ops = [c0, c1, a]
    trunc = [int(rng.integers(3, 8))]

    expr_out_of_order = Dagger(c1) * c0 * Dagger(a)
    expr_reordered = -c0 * Dagger(c1) * Dagger(a)

    mat_out_of_order = hilbert_space.to_matrix(
        expr_out_of_order, ops, sparse=sparse, truncation=trunc
    )[sympy.S.One]
    mat_reordered = hilbert_space.to_matrix(
        expr_reordered, ops, sparse=sparse, truncation=trunc
    )[sympy.S.One]

    np.testing.assert_allclose(as_dense(mat_out_of_order), as_dense(mat_reordered))


@pytest.mark.parametrize("sparse", [False, True])
def test_mixed_number_operator_commutes_with_bosons(sparse, rng, as_dense):
    """Verify [n_c, a] = 0 with n_c = c^† c by comparing n_c a and a n_c for operator list [c, a]."""
    c = FermionOp("c")
    a = BosonOp("a")
    ops = [c, a]
    trunc = int(rng.integers(3, 8))

    n_c = Dagger(c) * c

    expr_left = n_c * a
    expr_right = a * n_c

    mat_left = hilbert_space.to_matrix(expr_left, ops, sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]
    mat_right = hilbert_space.to_matrix(
        expr_right, ops, sparse=sparse, truncation=trunc
    )[sympy.S.One]

    np.testing.assert_allclose(as_dense(mat_left), as_dense(mat_right))


@pytest.mark.parametrize("sparse", [False, True])
def test_mixed_boson_decomposition_block(sparse, rng):
    """Bosonic block from mixed matrix should decompose to expected coefficients."""
    c = FermionOp("c")
    a = BosonOp("a")
    ops = [c, a]
    trunc = int(rng.integers(3, 7))

    expr = 1.5 + 2.0 * Dagger(a) * a
    matrix = hilbert_space.to_matrix(expr, ops, sparse=sparse, truncation=trunc)[
        sympy.S.One
    ]

    pauli_blocks = fermions._pauli_coefficients(matrix, 1)
    assert set(pauli_blocks.keys()) == {"I"}

    decomposition = bosons.BosonDecomposition(pauli_blocks["I"], truncation=trunc)
    assert decomposition[((0, 0),)] == pytest.approx(1.5)
    assert decomposition[((1, 1),)] == pytest.approx(2.0)


@pytest.mark.parametrize("sparse", [False, True])
def test_jaynes_cummings_round_trip(sparse, as_dense):
    """Mixed inversion should preserve a Jaynes-Cummings Hamiltonian."""
    c = FermionOp("c")
    a = BosonOp("a")
    operators = [c, a]
    truncation = 6

    hamiltonian = (
        -0.5 + Dagger(c) * c + Dagger(a) * a + 0.2 * (Dagger(c) * a + c * Dagger(a))
    )
    matrix = hilbert_space.to_matrix(
        hamiltonian,
        operators,
        sparse=sparse,
        truncation=truncation,
    )[sympy.S.One]
    recovered = hilbert_space.to_operators(matrix, operators, truncation=truncation)
    recovered_matrix = hilbert_space.to_matrix(
        recovered,
        operators,
        sparse=sparse,
        truncation=truncation,
    )[sympy.S.One]

    np.testing.assert_allclose(as_dense(recovered_matrix), as_dense(matrix))


@pytest.mark.parametrize("sparse", [False, True])
def test_interleaved_basis_round_trip(sparse, as_dense):
    """Both conversion directions use the stable fermion-then-boson partition."""
    c0, c1 = sympy.symbols("c0:2", cls=FermionOp)
    a, b = sympy.symbols("a b", cls=BosonOp)
    interleaved = [a, c0, b, c1]
    canonical = [c0, c1, a, b]
    expression = (
        sympy.Rational(1, 3)
        + Dagger(c1) * c0 * Dagger(a) * b
        + sympy.Rational(1, 5) * Dagger(c0) * c0 * Dagger(b) * b
    )

    matrix = hilbert_space.to_matrix(
        expression, interleaved, sparse=sparse, truncation=[2, 1]
    )[sympy.S.One]
    canonical_matrix = hilbert_space.to_matrix(
        expression, canonical, sparse=sparse, truncation=[2, 1]
    )[sympy.S.One]
    recovered = hilbert_space.to_operators(matrix, interleaved, truncation=[2, 1])

    np.testing.assert_allclose(as_dense(matrix), as_dense(canonical_matrix))
    assert sympy.expand(recovered - expression) == 0
