import numpy as np
import scipy
import pytest
import sympy
import random
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

from second_quantization import hilbert_space

# Fixtures


@pytest.fixture(scope="module", params=[2, 3, 4, 5, 6])
def get_n(request):
    """
    Return a list of orders to compute.
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
    Return a set of tuples that are meant to pair up combinations of ops
    """
    return request.param


# Builders


def get_list_of_operators(get_n):
    return [FermionOp("c_{}".format(i)) for i in range(get_n)]


def get_list_of_reals(get_n):
    return sympy.symbols(["a_{}".format(i) for i in range(get_n)], real=True)


def get_diagonal_term(ops, symbs):
    return sum([s * Dagger(op) * op for s, op in zip(symbs, ops)])


def get_random_hopping(ops, indices):
    if isinstance(indices, list):
        return [get_random_hopping(ops, pair) for pair in indices]
    i, j = indices
    return Dagger(ops[i]) * ops[j] + Dagger(ops[j]) * ops[i]


def get_random_pairing(ops, indices):
    if isinstance(indices, list):
        return [get_random_pairing(ops, pair) for pair in indices]
    i, j = indices
    return Dagger(ops[i]) * Dagger(ops[j]) + ops[j] * ops[i]


def build_random_hamiltonian(get_n, get_combinations):
    ops = get_list_of_operators(get_n)
    symbs = get_list_of_reals(get_n + len(get_combinations))
    diag_symbs = symbs[:get_n]
    hopping_symbs = symbs[get_n:]

    diag = get_diagonal_term(ops, diag_symbs)
    hopping = get_random_hopping(ops, hopping_symbs)
    supercond = get_random_pairing(ops, hopping_symbs)

    hamiltonian = (
        diag
        + sum([s * hop for s, hop in zip(hopping_symbs, hopping)])
        + sum([s * pair for s, pair in zip(hopping_symbs, supercond)])
    )
    return hamiltonian


# Tests


# Input and output types
def test_output_types(get_n):
    list_of_fermions = get_list_of_operators(get_n)
    basis_ops_sparse = hilbert_space.basis_operators(list_of_fermions, sparse=True)
    basis_ops_dense = hilbert_space.basis_operators(list_of_fermions, sparse=False)
    assert np.all([scipy.sparse.issparse(v) for v in basis_ops_sparse.values()])
    assert np.all([isinstance(v, np.ndarray) for v in basis_ops_dense.values()])
    ordered_keys_dense = sympy.ordered([k for k in basis_ops_dense.keys()])
    ordered_keys_sparse = sympy.ordered([k for k in basis_ops_dense.keys()])
    ordered_input = sympy.ordered(list_of_fermions)

    assert np.all([k1 == k2 for k1, k2 in zip(ordered_keys_dense, ordered_input)])
    assert np.all([k1 == k2 for k1, k2 in zip(ordered_keys_sparse, ordered_input)])
    return


# Idempotency


def test_idempotency():
    fermions = c_0, c_1, c_2 = sympy.symbols("c_:3", cls=FermionOp)
    expr = Dagger(c_1) * c_0 * c_2
    mat = hilbert_space.to_matrix(expr, fermions, sparse=False)
    ops = hilbert_space.to_operators(mat, fermions)
    assert normal_ordered_form(ops) == normal_ordered_form(expr)


def test_single_operator():
    c = FermionOp("c")
    mat = hilbert_space.to_matrix(c, [c], sparse=False)
    ops = hilbert_space.to_operators(mat, [c])
    assert ops == c


def test_symbolic_basis_matrix_action():
    """
    Test that the symbolic basis elements correctly map to their corresponding
    positions in the Hilbert space when converted to matrices and applied to vacuum.

    Steps:
    1. Generate the symbolic basis using symbolic_basis
    2. Choose a random element from the basis
    3. Extract operators and convert to matrix using to_matrix
    4. Apply matrix to vacuum state
    5. Verify the result has non-zero value only at the expected position
    """
    # Test with different numbers of fermions
    for n_fermions in [2, 3, 4]:
        # Create fermionic operators
        fermions = [FermionOp(f"c_{i}") for i in range(n_fermions)]

        # Generate symbolic basis
        symbolic_states = hilbert_space.symbolic_basis(fermions)
        print(symbolic_states)

        # The vacuum state is always the first element (all zeros)
        vacuum_state = np.zeros(2**n_fermions)
        vacuum_state[0] = 1

        # Test multiple random elements
        for _ in range(min(5, len(symbolic_states))):  # Test up to 5 random elements
            # Choose a random element from the symbolic basis
            random_index = random.randint(0, len(symbolic_states) - 1)
            random_element = symbolic_states[random_index]

            # Skip the vacuum state (S.One) as it's trivial
            if random_element == sympy.S.One:
                continue

            # Convert the symbolic element to matrix representation
            matrix_dict = hilbert_space.to_matrix(
                random_element, fermions, sparse=False
            )

            # Since random_element should be a product of creation operators,
            # it should have coefficient 1 (S.One as key)
            assert sympy.S.One in matrix_dict
            operator_matrix = matrix_dict[sympy.S.One]

            # Apply the matrix to the vacuum state
            _vacuum_state = vacuum_state.copy()
            result_state = operator_matrix @ _vacuum_state

            # Find non-zero elements

            non_zero_indices = np.where(np.abs(result_state) > 1e-10)[0]
            result_index = non_zero_indices[0]
            print(matrix_dict)
            print(result_state)
            print(random_element)
            print(operator_matrix)
            print(random_index)
            print(result_index)

            # There should be exactly one non-zero element
            assert (
                len(non_zero_indices) == 1
            ), f"Expected exactly one non-zero element, got {len(non_zero_indices)}"

            # The non-zero element should be at the position corresponding to the random element

            assert (
                result_index == random_index
            ), f"Expected non-zero at index {random_index}, got {result_index}"

            # The value should be 1 (up to sign, depending on fermion ordering)
            assert (
                abs(abs(result_state[result_index]) - 1.0) < 1e-10
            ), f"Expected |amplitude| = 1, got {abs(result_state[result_index])}"


def test_symbolic_basis_properties():
    """
    Test basic properties of the symbolic basis.
    """
    # Test with 3 fermions
    fermions = [FermionOp(f"c_{i}") for i in range(3)]
    symbolic_states = hilbert_space.symbolic_basis(fermions)

    # Should have 2^n states
    assert len(symbolic_states) == 2**3

    # First state should be vacuum (S.One)
    assert symbolic_states[0] == sympy.S.One

    # Check some specific states
    # State |001⟩ should correspond to c_0†|0⟩
    assert symbolic_states[1] == Dagger(fermions[2])

    # State |010⟩ should correspond to c_1†|0⟩
    assert symbolic_states[2] == Dagger(fermions[1])

    # State |011⟩ should correspond to c_1†c_0†|0⟩
    expected_state_3 = Dagger(fermions[1]) * Dagger(fermions[2])
    assert symbolic_states[3] == expected_state_3


def test_string_basis_properties():
    """
    Test the string basis generation.
    """
    fermions = [FermionOp(f"c_{i}") for i in range(3)]
    strings = hilbert_space.string_basis(fermions)

    # Should have 2^n strings
    assert len(strings) == 2**3

    # Check specific strings
    assert strings[0] == "000"  # vacuum
    assert strings[1] == "001"  # first fermion occupied
    assert strings[2] == "010"  # second fermion occupied
    assert strings[3] == "011"  # first and second fermions occupied
    assert strings[7] == "111"  # all fermions occupied
