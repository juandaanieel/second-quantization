from functools import reduce

import numpy as np
import pytest
import scipy
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from second_quantization import fermions, hilbert_space


@pytest.mark.parametrize("sparse", [False, True])
def test_fermion_creation_anticommutation_sign(sparse, as_dense):
    """Check c_i^† c_j^† = - c_j^† c_i^† for i≠j to enforce {c_i^†, c_j^†}=0."""
    c0, c1 = sympy.symbols("c0:2", cls=FermionOp)
    ops = [c0, c1]

    expr_12 = Dagger(c1) * Dagger(c0)
    expr_21 = Dagger(c0) * Dagger(c1)

    mat_12 = hilbert_space.to_matrix(expr_12, ops, sparse=sparse)[sympy.S.One]
    mat_21 = hilbert_space.to_matrix(expr_21, ops, sparse=sparse)[sympy.S.One]

    np.testing.assert_allclose(as_dense(mat_12), -as_dense(mat_21))


@pytest.mark.parametrize("sparse", [False, True])
def test_fermion_number_anticommutator_identity(sparse, as_dense):
    """Verify {c, c^†} = 1 so c c^† + c^† c yields the identity matrix."""
    c = FermionOp("c")
    ops = [c]

    expr = c * Dagger(c) + Dagger(c) * c
    mat = hilbert_space.to_matrix(expr, ops, sparse=sparse)[sympy.S.One]

    np.testing.assert_allclose(as_dense(mat), np.eye(2))


@pytest.mark.parametrize("sparse", [False, True])
def test_fermion_mixed_order_against_reference_list(sparse, as_dense):
    """Confirm c_j^† c_i = - c_i c_j^† (i≠j), even when operator order differs from the reference list."""
    c0, c1, c2 = sympy.symbols("c0:3", cls=FermionOp)
    ops = [c0, c1, c2]

    expr_out_of_order = Dagger(c2) * c0
    expr_reordered = -c0 * Dagger(c2)

    mat_out_of_order = hilbert_space.to_matrix(expr_out_of_order, ops, sparse=sparse)[
        sympy.S.One
    ]
    mat_reordered = hilbert_space.to_matrix(expr_reordered, ops, sparse=sparse)[
        sympy.S.One
    ]

    np.testing.assert_allclose(as_dense(mat_out_of_order), as_dense(mat_reordered))


@pytest.fixture(scope="module", params=["IIII", "XZXY", "Y", "IIYZ"])
def pauli_string(request):
    return request.param


def build_from_string(pauli_string):
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }
    return reduce(np.kron, (matrices[letter] for letter in pauli_string))


def test_pauli_string(pauli_string):
    decomposition = fermions.PauliDecomposition(build_from_string(pauli_string))
    assert set(decomposition) == {pauli_string}
    assert np.isclose(decomposition[pauli_string], 1.0)


def test_pauli_output(pauli_string):
    decomposition = fermions.PauliDecomposition(build_from_string(pauli_string))
    assert isinstance(decomposition, dict)
    assert all(isinstance(key, str) for key in decomposition)
    assert all(isinstance(value, complex | float) for value in decomposition.values())


def test_pauli_decomposition_invalid_shapes():
    with pytest.raises(ValueError):
        fermions.PauliDecomposition(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        fermions.PauliDecomposition(np.eye(3))


def test_pauli_decomposition_sparse_and_sympy_inputs():
    sparse_eye = scipy.sparse.eye(4, format="csr")
    assert fermions.PauliDecomposition(sparse_eye) == {"II": 1.0}

    decomposition = fermions.PauliDecomposition(sympy.eye(2))
    assert decomposition == {"I": 1}


def test_pauli_decomposition_threshold():
    matrix = 1e-12 * build_from_string("X")
    assert fermions.PauliDecomposition(matrix, threshold=1e-10) == {}


@pytest.mark.parametrize("label", ["", "IY"])
@pytest.mark.parametrize(
    "coefficient, threshold, expected",
    [
        (sympy.Symbol("x"), 0.0, sympy.Symbol("x")),
        (sympy.Symbol("x"), 1e-10, sympy.Symbol("x")),
        (sympy.Rational(1, 10**400), 0.0, sympy.Rational(1, 10**400)),
        (sympy.Rational(1, 10**400), 1e-10, 0),
        (sympy.Rational(1, 3) + sympy.I / 7, 0.0, sympy.Rational(1, 3) + sympy.I / 7),
        (sympy.S.Zero, 0.0, 0),
    ],
)
def test_pauli_decomposition_symbolic_coefficients(
    label, coefficient, threshold, expected
):
    matrix = sympy.ones(1)
    if label:
        matrix = sympy.kronecker_product(
            sympy.eye(2), sympy.Matrix([[0, -sympy.I], [sympy.I, 0]])
        )
    decomposition = fermions.PauliDecomposition(coefficient * matrix, threshold)
    assert decomposition == ({label: expected} if expected != 0 else {})


def test_to_operators_small_coefficient():
    c = FermionOp("c")
    coefficient = 1e-30
    expression = hilbert_space.to_operators(coefficient * np.eye(2), [c])
    np.testing.assert_allclose(complex(expression), coefficient, rtol=1e-12, atol=0)


@pytest.mark.parametrize("coefficient", [sympy.Symbol("x"), sympy.Rational(1, 10**400)])
def test_to_operators_exact_coefficient(coefficient):
    expression = hilbert_space.to_operators(
        coefficient * sympy.eye(2), [FermionOp("c")]
    )
    assert expression == coefficient
