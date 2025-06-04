import numpy as np
import scipy
import pytest
import sympy
from functools import reduce
from operator import mul
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from second_quantization import hilbert_space

# Fixtures


@pytest.fixture(scope="module", params=[2, 3, 4, 5, 6])
def get_n(request):
    """
    Fixture that provides different sizes for the number of fermionic operators to test with.

    Yields:
        int: Number of fermionic operators.
    """
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        (0, 1),
        [(0, 1), (1, 2)],
        (0, 1),
        [(0, 1), (1, 2), (0, 2)],
        [(1, 2), (2, 3), (3, 4), (0, 5), (1, 4)],
    ],
)
def get_combinations(request):
    """
    Fixture that provides various index combinations for hopping or pairing terms.

    Yields:
        Union[Tuple[int, int], List[Tuple[int, int]]]: Index pairs for operator interactions.
    """
    return request.param


# Builders


def get_list_of_operators(get_n):
    """
    Generate a list of Fermion creation/annihilation operators.

    Args:
        get_n (int): Number of operators to generate.

    Returns:
        List[FermionOp]: List of Fermion operators.
    """
    return [FermionOp(f"c_{i}") for i in range(get_n)]


def get_list_of_reals(get_n):
    """
    Generate a list of real-valued symbolic coefficients.

    Args:
        get_n (int): Number of symbols to generate.

    Returns:
        Tuple[sympy.Symbol]: Tuple of real-valued sympy symbols.
    """
    return sympy.symbols([f"a_{i}" for i in range(get_n)], real=True)


def get_diagonal_term(ops, symbs):
    """
    Construct a diagonal Hamiltonian term in second quantization.

    Args:
        ops (List[FermionOp]): List of Fermion operators.
        symbs (List[sympy.Symbol]): Coefficients for each diagonal term.

    Returns:
        Expr: Diagonal Hamiltonian expression.
    """
    return sum([s * Dagger(op) * op for s, op in zip(symbs, ops)])


def get_random_hopping(ops, indices):
    """
    Construct hopping terms between specified operator indices.

    Args:
        ops (List[FermionOp]): List of Fermion operators.
        indices (Union[Tuple[int, int], List[Tuple[int, int]]]): Index pairs for hopping.

    Returns:
        Union[Expr, List[Expr]]: Hopping term(s).
    """
    if isinstance(indices, list):
        return [get_random_hopping(ops, pair) for pair in indices]
    i, j = indices
    return Dagger(ops[i]) * ops[j] + Dagger(ops[j]) * ops[i]


def get_random_pairing(ops, indices):
    """
    Construct pairing terms between specified operator indices.

    Args:
        ops (List[FermionOp]): List of Fermion operators.
        indices (Union[Tuple[int, int], List[Tuple[int, int]]]): Index pairs for pairing.

    Returns:
        Union[Expr, List[Expr]]: Pairing term(s).
    """
    if isinstance(indices, list):
        return [get_random_pairing(ops, pair) for pair in indices]
    i, j = indices
    return Dagger(ops[i]) * Dagger(ops[j]) + ops[j] * ops[i]


def build_random_hamiltonian(get_n, get_combinations):
    """
    Construct a random Hamiltonian with diagonal, hopping, and pairing terms.

    Args:
        get_n (int): Number of Fermion operators.
        get_combinations (List[Tuple[int, int]]): Index pairs for hopping/pairing.

    Returns:
        Expr: Full Hamiltonian expression.
    """
    ops = get_list_of_operators(get_n)
    symbs = get_list_of_reals(get_n + len(get_combinations))
    diag_symbs = symbs[:get_n]
    hopping_symbs = symbs[get_n:]

    diag = get_diagonal_term(ops, diag_symbs)
    hopping = get_random_hopping(ops, get_combinations)
    supercond = get_random_pairing(ops, get_combinations)

    hamiltonian = (
        diag
        + sum([s * hop for s, hop in zip(hopping_symbs, hopping)])
        + sum([s * pair for s, pair in zip(hopping_symbs, supercond)])
    )
    return hamiltonian


# Tests


def test_output_types(get_n):
    """
    Verify that `basis_operators` returns correct types for sparse and dense formats.
    """
    list_of_fermions = get_list_of_operators(get_n)
    basis_ops_sparse, _ = hilbert_space.basis_operators(list_of_fermions, sparse=True)
    basis_ops_dense, _ = hilbert_space.basis_operators(list_of_fermions, sparse=False)

    assert np.all([scipy.sparse.issparse(v) for v in basis_ops_sparse.values()])
    assert np.all([isinstance(v, np.ndarray) for v in basis_ops_dense.values()])

    ordered_keys_dense = sympy.ordered(basis_ops_dense.keys())
    ordered_keys_sparse = sympy.ordered(basis_ops_sparse.keys())
    ordered_input = sympy.ordered(list_of_fermions)

    assert np.all([k1 == k2 for k1, k2 in zip(ordered_keys_dense, ordered_input)])
    assert np.all([k1 == k2 for k1, k2 in zip(ordered_keys_sparse, ordered_input)])


def test_idempotency():
    """
    Ensure that converting an operator expression to matrix and back preserves the expression.
    """
    fermions = c_0, c_1, c_2 = sympy.symbols("c_:3", cls=FermionOp)
    expr = Dagger(c_1) * c_0 * c_2
    mat = hilbert_space.to_matrix(expr, fermions, sparse=False)
    ops = hilbert_space.to_operators(mat, fermions)
    assert normal_ordered_form(ops) == normal_ordered_form(expr)


def test_single_operator():
    """
    Check that a single Fermion operator is preserved through matrix conversion.
    """
    c = FermionOp("c")
    mat = hilbert_space.to_matrix(c, [c], sparse=False)
    ops = hilbert_space.to_operators(mat, [c])
    assert ops == c


def test_combinations(get_n):
    """
    Test that all combinations of a subset of Fermion operators are correctly generated.
    """
    # Create a list of Fermion operators for testing
    list_of_fermions = [FermionOp(f"c_{i}") for i in range(get_n)]

    # Take the first half of the operators as a subset
    subset = list_of_fermions[: get_n // 2]

    # Keep reference to all operators
    all_operators = list_of_fermions

    # Use dense matrices for this test
    sparse = False

    # Generate all possible combinations of the subset operators
    all_combinations = hilbert_space._get_all_combinations(
        subset, all_operators, sparse
    )

    # Test that individual operators from the subset are included
    assert subset[0] in all_combinations

    # Test that the product of all operators in the subset is included
    assert reduce(mul, subset) in all_combinations

    # Test combinations of non-adjacent operators (if subset has more than 1 element)
    if get_n // 2 > 1:
        assert reduce(mul, [subset[0], subset[-1]]) in all_combinations


def test_partial_trace(get_n):
    """
    Validate partial trace by checking reduced density matrix properties for a bipartite state.
    """
    # Create a list of fermion operators for the full system
    list_of_fermions = [FermionOp(f"c_{i}") for i in range(get_n)]

    # Generate a random probability amplitude for the first basis state
    p_0 = np.random.uniform(0, 1)

    # Create a state vector with non-zero amplitudes only in first and last basis states
    # This represents a superposition |0...0⟩ + |1...1⟩
    v = np.array([p_0, *(0 for _ in range(2 ** len(list_of_fermions) - 2)), 1 - p_0])

    # Normalize the state vector
    v = v / np.sqrt(v.conj().T @ v)

    # Create the density matrix from the pure state
    m = np.outer(v, v)

    # Generate partial trace operators for tracing out all but the first fermion
    # This will give us the reduced density matrix of the first fermion only
    p = hilbert_space.partial_trace_generators(
        list_of_fermions[1:], list_of_fermions, False
    )

    # Compute the reduced density matrix by applying partial trace
    # Sum over all basis states of the traced-out subsystem
    m_red = np.sum(
        [p[:, i, :] @ m @ p[:, i, :].conj().T for i in range(p.shape[1])], axis=0
    )

    # Verify that the (0,0) element is preserved under partial trace
    assert np.isclose(m_red[0, 0], m[0, 0])

    # Verify that the reduced density matrix has unit trace (probability conservation)
    assert np.isclose(np.trace(m_red), 1)
