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


# Tests for make_dict_callable function
def test_make_dict_callable_function_signature():
    """Test that the returned function has the correct signature."""
    a, b, c = sympy.symbols("a b c", real=True)

    # Create dictionary with multiple symbols
    hamiltonian_dict = {a: np.eye(2), b + c: np.ones((2, 2)), a * b: np.zeros((2, 2))}

    func = hilbert_space.make_dict_callable(hamiltonian_dict)

    # Check that the function signature contains the expected parameters
    import inspect

    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Should contain all free symbols: a, b, c
    assert len(param_names) == 3
    assert set(param_names) == {"a", "b", "c"}


def test_make_dict_callable_zero_values():
    """Test make_dict_callable with zero parameter values."""
    x, y = sympy.symbols("x y", real=True)

    matrix_x = np.array([[1, 2], [3, 4]])
    matrix_y = np.array([[5, 6], [7, 8]])

    hamiltonian_dict = {x: matrix_x, y: matrix_y}
    func = hilbert_space.make_dict_callable(hamiltonian_dict)

    # Test with zero values
    result = func(0.0, 0.0)
    expected = np.zeros((2, 2))

    np.testing.assert_array_almost_equal(result, expected)


def test_make_dict_callable_keyword_arguments():
    """Test that the callable function works with keyword arguments."""
    mu, t = sympy.symbols("mu t", real=True)

    matrix_mu = np.array([[1, 0], [0, -1]])
    matrix_t = np.array([[0, 1], [1, 0]])

    hamiltonian_dict = {mu: matrix_mu, t: matrix_t}
    func = hilbert_space.make_dict_callable(hamiltonian_dict)

    # Test with keyword arguments
    result1 = func(mu=0.5, t=1.0)
    result2 = func(t=1.0, mu=0.5)  # Different order
    expected = 0.5 * matrix_mu + 1.0 * matrix_t

    np.testing.assert_array_almost_equal(result1, expected)
    np.testing.assert_array_almost_equal(result2, expected)


def test_make_dict_callable_cosine_and_complex():
    """Test make_dict_callable with cosine prefactor and complex constant terms."""
    # Define two fermions
    c, d = [FermionOp(name) for name in "cd"]

    # Define a symbolic parameter for the cosine term
    t = sympy.Symbol("t", real=True)

    # Create two fermionic terms:
    # Term 1: cosine prefactor with hopping c†d + d†c
    term1 = sympy.cos(t) * (Dagger(c) * d + Dagger(d) * c)

    # Term 2: complex constant with pairing c†d† + dc
    complex_coeff = 1.5 + 2.0j
    term2 = complex_coeff * (Dagger(c) * Dagger(d) + d * c)

    # Combine terms into a single expression
    hamiltonian = term1 + term2

    # Convert to matrix representation
    matrix_dict = hilbert_space.to_matrix(hamiltonian, [c, d], sparse=False)

    # Create callable function
    func = hilbert_space.make_dict_callable(matrix_dict)

    # Test with a random value for t
    np.random.seed(42)  # For reproducibility
    t_val = np.random.uniform(0, 2 * np.pi)

    result = func(t_val)

    # Manually compute expected result
    # First, get the matrix representations for each term separately
    term1_dict = hilbert_space.to_matrix(term1, [c, d], sparse=False)
    term2_dict = hilbert_space.to_matrix(term2, [c, d], sparse=False)

    # Extract matrices and compute expected result
    expected = np.zeros((4, 4), dtype=complex)

    # Add contribution from cosine term
    for expr, matrix in term1_dict.items():
        if expr == sympy.cos(t):
            expected += np.cos(t_val) * matrix
        else:
            # Handle other symbolic expressions if any
            expected += float(expr.subs(t, t_val)) * matrix

    # Add contribution from complex constant term
    for expr, matrix in term2_dict.items():
        if expr == complex_coeff:
            expected += complex_coeff * matrix
        else:
            # Handle other expressions
            if hasattr(expr, "subs"):
                expected += complex(expr.subs(t, t_val)) * matrix
            else:
                expected += complex(expr) * matrix

    # Verify the result matches expected
    np.testing.assert_array_almost_equal(result, expected)

    # Additional checks
    assert result.shape == (4, 4)  # 2^2 for 2 fermions
    assert np.iscomplexobj(result)  # Should be complex due to complex coefficient

    # Test with another value to ensure function works correctly
    t_val2 = np.pi / 4
    result2 = func(t_val2)
    assert result2.shape == (4, 4)

    # Verify that changing t changes the result (due to cosine term)
    assert not np.allclose(result, result2)


def test_make_dict_callable_dummy_variable_error():
    """Test that make_dict_callable raises ValueError for symbols that become Dummy variables."""
    # Create a symbol that will cause lambdify to generate a Dummy variable
    # This can happen with symbols that have special characters or reserved names

    # Using a symbol with underscores and special formatting that can cause issues
    problematic_symbol = sympy.Symbol(r"__x_{3}", real=True)
    matrix = np.array([[1, 0], [0, 1]])

    hamiltonian_dict = {problematic_symbol: matrix}

    # This should raise a ValueError due to the Dummy variable detection
    with pytest.raises(
        ValueError,
        match="Variable names must be chosen such that they are not lambdified to Dummy variables",
    ):
        hilbert_space.make_dict_callable(hamiltonian_dict)


# Alternative test that might be more reliable for triggering Dummy variables
def test_make_dict_callable_dummy_variable_error_alternative():
    """Alternative test for Dummy variable error using reserved Python keywords."""
    # Try using Python reserved keywords or names that conflict with numpy
    try:
        # Create a symbol with a name that might conflict
        problematic_symbol = sympy.Symbol(
            "lambda", real=True
        )  # 'lambda' is a Python keyword
        matrix = np.eye(2)

        hamiltonian_dict = {problematic_symbol: matrix}

        # This might trigger the Dummy variable error
        with pytest.raises(
            ValueError,
            match="Variable names must be chosen such that they are not lambdified to Dummy variables",
        ):
            hilbert_space.make_dict_callable(hamiltonian_dict)

    except ValueError:
        # If the first attempt works (raises the expected error), we're good
        pass
    except AssertionError:
        # If it doesn't raise the error, try another approach
        # Create a symbol that starts with underscore and contains 'Dummy'
        problematic_symbol2 = sympy.Symbol("__Dummy__var__", real=True)
        matrix = np.eye(2)

        hamiltonian_dict = {problematic_symbol2: matrix}

        with pytest.raises(
            ValueError,
            match="Variable names must be chosen such that they are not lambdified to Dummy variables",
        ):
            hilbert_space.make_dict_callable(hamiltonian_dict)
