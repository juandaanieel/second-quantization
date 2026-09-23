import numpy as np
import scipy
import pytest
import sympy
import random
import math
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.boson import BosonOp

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


def _dims_for_case(ops, truncation):
    bosons = [op for op in ops if isinstance(op, BosonOp)]
    if truncation is None:
        trunc_list = []
    elif isinstance(truncation, int):
        trunc_list = [truncation] * len(bosons)
    else:
        trunc_list = list(truncation)

    dims = []
    boson_iter = iter(trunc_list)
    for op in ops:
        if isinstance(op, FermionOp):
            dims.append(2)
        else:
            dims.append(next(boson_iter) + 1)
    return dims


@pytest.fixture(
    scope="module",
    params=[
        {
            "label": "fermion",
            "ops": [FermionOp("c0"), FermionOp("c1")],
            "trunc": None,
            "expr": Dagger(FermionOp("c0")) * FermionOp("c1"),
        },
        {
            "label": "boson",
            "ops": [BosonOp("b0"), BosonOp("b1")],
            "trunc": [2, 3],
            "expr": Dagger(BosonOp("b0")) * BosonOp("b1"),
        },
        {
            "label": "mixed",
            "ops": [FermionOp("c0"), BosonOp("b0"), FermionOp("c1")],
            "trunc": [2],
            "expr": Dagger(FermionOp("c0")) * BosonOp("b0"),
        },
    ],
)
def operator_case(request):
    return request.param


# Builders


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


@pytest.mark.parametrize("sparse", [False, True])
def test_partial_trace_fermion_uses_symbolic_basis_sign(sparse):
    c0, c1, c2 = [FermionOp(f"c_{i}") for i in range(3)]
    operators = [c0, c1, c2]

    vacuum = np.zeros(8)
    vacuum[0] = 1
    pair = hilbert_space.to_matrix(Dagger(c0) * Dagger(c1), operators, sparse)[1]
    state = (vacuum + pair @ vacuum) / np.sqrt(2)
    density = np.outer(state, state.conj())

    p = hilbert_space.partial_trace_generators([c2], operators, sparse)
    reduced = np.sum(
        [p[:, i, :] @ density @ p[:, i, :].conj().T for i in range(p.shape[1])],
        axis=0,
    )

    expected = np.zeros((4, 4))
    expected[np.ix_([0, 3], [0, 3])] = 0.5
    np.testing.assert_allclose(reduced, expected)


def test_partial_trace_fermion_preserves_crossing_sign():
    c0, c1, c2 = [FermionOp(f"c_{i}") for i in range(3)]
    operators = [c0, c1, c2]

    vacuum = np.zeros(8)
    vacuum[0] = 1
    state = (
        hilbert_space.to_matrix(Dagger(c1), operators, False)[1] @ vacuum
        + hilbert_space.to_matrix(
            Dagger(c0) * Dagger(c1) * Dagger(c2), operators, False
        )[1]
        @ vacuum
    ) / np.sqrt(2)
    density = np.outer(state, state.conj())

    p = hilbert_space.partial_trace_generators([c1], operators, False)
    reduced = np.sum(
        [p[:, i, :] @ density @ p[:, i, :].conj().T for i in range(p.shape[1])],
        axis=0,
    )

    expected = np.zeros((4, 4))
    expected[np.ix_([0, 3], [0, 3])] = [[0.5, -0.5], [-0.5, 0.5]]
    np.testing.assert_allclose(reduced, expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_partial_trace_boson_maps_unequal_truncations(sparse):
    b0, b1, b2 = [BosonOp(f"b_{i}") for i in range(3)]
    operators = [b0, b1, b2]
    truncation = [1, 2, 1]

    state = np.zeros(12)
    state[4] = 1
    density = np.outer(state, state.conj())

    p = hilbert_space.partial_trace_generators(
        [b1], operators, sparse, truncation=truncation
    )
    reduced = np.sum(
        [p[:, i, :] @ density @ p[:, i, :].conj().T for i in range(p.shape[1])],
        axis=0,
    )

    expected = np.zeros((4, 4))
    expected[0, 0] = 1

    assert p.shape == (4, 3, 12)
    np.testing.assert_allclose(reduced, expected)


def test_partial_trace_bosons_only(rng):
    """Desired: tracing out a boson should preserve trace and use full bosonic dimension."""
    b0, b1 = [BosonOp(f"b_{i}") for i in range(2)]
    trunc = [2, 2]
    ops = [b0, b1]

    operator_dict = hilbert_space.basis_operators(ops, sparse=False, truncation=trunc)

    p = hilbert_space.partial_trace_generators(
        subset=[b1], all_operators=ops, sparse=False, operator_dict=operator_dict
    )

    dim_total = np.prod([t + 1 for t in trunc])
    v = rng.random(dim_total)
    v = v / np.linalg.norm(v)
    m = np.outer(v, v.conj())

    m_red = np.sum(
        [p[:, i, :] @ m @ p[:, i, :].conj().T for i in range(p.shape[1])], axis=0
    )

    dim_keep = trunc[0] + 1
    assert m_red.shape == (dim_keep, dim_keep)
    assert np.isclose(np.trace(m_red), 1.0)


def test_partial_trace_mixed_fermion_boson(rng):
    """Desired: tracing out boson in mixed system should yield valid fermionic density matrix."""
    c = FermionOp("c")
    b = BosonOp("b")
    trunc = [3]
    ops = [c, b]

    operator_dict = hilbert_space.basis_operators(ops, sparse=False, truncation=trunc)

    p = hilbert_space.partial_trace_generators(
        subset=[b], all_operators=ops, sparse=False, operator_dict=operator_dict
    )

    dim_total = 2 * (trunc[0] + 1)
    v = rng.random(dim_total)
    v = v / np.linalg.norm(v)
    m = np.outer(v, v.conj())

    m_red = np.sum(
        [p[:, i, :] @ m @ p[:, i, :].conj().T for i in range(p.shape[1])], axis=0
    )

    assert m_red.shape == (2, 2)
    assert np.isclose(np.trace(m_red), 1.0)


@pytest.mark.parametrize("sparse", [False, True])
def test_partial_trace_mixed_out_of_order_expression(sparse):
    c0, c1 = [FermionOp(f"c_{i}") for i in range(2)]
    b = BosonOp("b")
    operators = [c0, b, c1]
    truncation = [1]

    vacuum = np.zeros(8)
    vacuum[0] = 1
    initial = (
        hilbert_space.to_matrix(Dagger(c0), operators, sparse, truncation=truncation)[1]
        @ vacuum
    )

    expressions = [
        Dagger(c1) * c0 * Dagger(b),
        -c0 * Dagger(c1) * Dagger(b),
    ]
    p = hilbert_space.partial_trace_generators(
        [b], operators, sparse, truncation=truncation
    )
    reduced = []
    for expression in expressions:
        state = (
            hilbert_space.to_matrix(
                expression, operators, sparse, truncation=truncation
            )[1]
            @ initial
        )
        density = np.outer(state, state.conj())
        reduced.append(
            np.sum(
                [p[:, i, :] @ density @ p[:, i, :].conj().T for i in range(p.shape[1])],
                axis=0,
            )
        )

    expected = np.zeros((4, 4))
    expected[1, 1] = 1
    np.testing.assert_allclose(reduced[0], reduced[1])
    np.testing.assert_allclose(reduced[0], expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_partial_trace_interleaved_remaining_modes(sparse):
    c = FermionOp("c")
    a, b = sympy.symbols("a b", cls=BosonOp)
    state = np.arange(12) + 1j * np.arange(12)[::-1]
    state /= np.linalg.norm(state)
    density = np.outer(state, state.conj())
    p = hilbert_space.partial_trace_generators(
        [b], [a, c, b], sparse=sparse, truncation=[2, 1]
    )
    reduced = sum(p[:, i, :] @ density @ p[:, i, :].conj().T for i in range(p.shape[1]))
    expected = np.trace(density.reshape(6, 2, 6, 2), axis1=1, axis2=3)
    np.testing.assert_allclose(reduced, expected)


def test_basis_and_parity_operator_mixed_cases(operator_case):
    ops = operator_case["ops"]
    trunc = operator_case["trunc"]
    dims = _dims_for_case(ops, trunc)
    total_dim = int(np.prod(dims))

    fermion_dict, boson_dict = hilbert_space.basis_operators(
        ops, sparse=False, truncation=trunc
    )
    assert 1 in fermion_dict
    assert 1 in boson_dict

    parity_dense = hilbert_space.parity_operator(ops, sparse=False, truncation=trunc)
    parity_sparse = hilbert_space.parity_operator(ops, sparse=True, truncation=trunc)

    assert parity_dense.shape == (total_dim, total_dim)
    np.testing.assert_array_equal(parity_dense, parity_sparse.toarray())


def test_to_matrix_mixed_cases(operator_case):
    ops = operator_case["ops"]
    trunc = operator_case["trunc"]
    dims = _dims_for_case(ops, trunc)
    total_dim = int(np.prod(dims))
    expr = operator_case["expr"]

    operator_dict = hilbert_space.basis_operators(ops, sparse=False, truncation=trunc)
    mats_default = hilbert_space.to_matrix(expr, ops, sparse=False, truncation=trunc)
    mats_cached = hilbert_space.to_matrix(
        expr, ops, sparse=False, truncation=trunc, operator_dict=operator_dict
    )

    assert mats_default.keys() == mats_cached.keys()
    for key in mats_default:
        np.testing.assert_allclose(mats_default[key], mats_cached[key])
        assert mats_default[key].shape == (total_dim, total_dim)


@pytest.mark.parametrize("sparse", [False, True])
def test_numeric_matrix_round_trip(operator_case, sparse, as_dense):
    ops = operator_case["ops"]
    truncation = operator_case["trunc"]
    dimension = int(np.prod(_dims_for_case(ops, truncation)))
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    supplied = scipy.sparse.csr_array(matrix) if sparse else matrix
    expression = hilbert_space.to_operators(supplied, ops, truncation=truncation)
    recovered = hilbert_space.to_matrix(
        expression, ops, sparse=sparse, truncation=truncation
    )
    actual = sum(complex(key) * as_dense(value) for key, value in recovered.items())
    np.testing.assert_allclose(actual, matrix, rtol=1e-12, atol=1e-12)


def test_symbolic_basis_and_powers_mixed_cases(operator_case):
    ops = operator_case["ops"]
    trunc = operator_case["trunc"]
    dims = _dims_for_case(ops, trunc)
    total_dim = int(np.prod(dims))
    trunc_arg = trunc if trunc is not None else 1

    powers = hilbert_space._generate_all_powers(ops, trunc_arg)
    assert len(powers) == total_dim

    symbols = hilbert_space.symbolic_basis(ops, truncation=trunc_arg)
    assert len(symbols) == total_dim
    assert symbols[0] == sympy.S.One


def test_make_dict_callable_mixed_cases(operator_case):
    ops = operator_case["ops"]
    trunc = operator_case["trunc"]
    expr = operator_case["expr"]
    dims = _dims_for_case(ops, trunc)
    total_dim = int(np.prod(dims))

    mat_dict = hilbert_space.to_matrix(expr, ops, sparse=False, truncation=trunc)
    func = hilbert_space.make_dict_callable(mat_dict)

    result = func() if func.__signature__.parameters else func
    out = result if isinstance(result, np.ndarray) else func()
    assert isinstance(out, np.ndarray)
    assert out.shape == (total_dim, total_dim)


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
        symbolic_states = hilbert_space.symbolic_basis(fermions, truncation=1)
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


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_symbolic_basis_mixed_matrix_action(sparse, interleaved):
    c0, c1 = sympy.symbols("c0:2", cls=FermionOp)
    a, b = sympy.symbols("a b", cls=BosonOp)
    operators = [a, c0, b, c1] if interleaved else [c0, c1, a, b]
    states = hilbert_space.symbolic_basis(operators, truncation=[2, 1])
    identity = np.eye(24)

    for index, state in enumerate(states):
        occupations = np.unravel_index(index, (2, 2, 3, 2))
        norm = math.sqrt(math.prod(math.factorial(n) for n in occupations[2:]))
        matrix = hilbert_space.to_matrix(
            state, operators, sparse=sparse, truncation=[2, 1]
        )[1]
        np.testing.assert_allclose(matrix @ identity[0], norm * identity[index])


@pytest.mark.parametrize("truncation", [None, [2, 1]])
def test_symbolic_basis_invalid_truncation(truncation):
    with pytest.raises(ValueError):
        hilbert_space.symbolic_basis([BosonOp("b")], truncation=truncation)


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


def test_make_dict_callable_output_types():
    """Test make_dict_callable random matrices for their output types."""
    x, y = sympy.symbols("x y", real=True)

    matrix_x = np.random.random((50, 50))
    matrix_y = np.random.random((50, 50))

    hamiltonian_dict = {x: matrix_x, y: matrix_y}
    func = hilbert_space.make_dict_callable(hamiltonian_dict)

    assert isinstance(func(*np.random.random(2)), np.ndarray)

    hamiltonian_dict = {
        x: scipy.sparse.csr_array(matrix_x),
        y: scipy.sparse.csr_array(matrix_y),
    }
    func = hilbert_space.make_dict_callable(hamiltonian_dict)

    assert scipy.sparse.issparse(func(*np.random.random(2)))


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

    # term1_dict = hilbert_space.to_matrix(term1, [c, d], sparse=False)
    # term2_dict = hilbert_space.to_matrix(term2, [c, d], sparse=False)

    # Extract matrices and compute expected result
    expected = np.zeros_like(result)
    for prefactor, matrix in matrix_dict.items():
        value = (
            complex(prefactor.subs(t, t_val))
            if hasattr(prefactor, "subs")
            else complex(prefactor)
        )
        expected += value * matrix

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


def test_parity_operator_output_types():
    """Test that parity_operator returns the correct output types for sparse and dense matrices."""
    # Test with different numbers of operators
    for n_ops in [1, 2, 3, 4]:
        operators = [FermionOp(f"c_{i}") for i in range(n_ops)]

        # Test dense output
        parity_dense = hilbert_space.parity_operator(operators, sparse=False)
        assert isinstance(
            parity_dense, np.ndarray
        ), f"Dense parity operator should return numpy array for {n_ops} operators"
        assert (
            parity_dense.ndim == 2
        ), f"Parity operator should be 2D matrix for {n_ops} operators"
        assert parity_dense.shape == (
            2**n_ops,
            2**n_ops,
        ), f"Shape should be ({2**n_ops}, {2**n_ops}) for {n_ops} operators"

        # Test sparse output
        parity_sparse = hilbert_space.parity_operator(operators, sparse=True)
        assert isinstance(
            parity_sparse, scipy.sparse._csr.csr_matrix
        ), f"Sparse parity operator should return CSR array for {n_ops} operators"
        assert parity_sparse.shape == (
            2**n_ops,
            2**n_ops,
        ), f"Shape should be ({2**n_ops}, {2**n_ops}) for {n_ops} operators"

        np.testing.assert_array_equal(
            parity_dense,
            parity_sparse.toarray(),
            err_msg=f"Sparse and dense parity operators should have same values for {n_ops} operators",
        )


def test_parity_operator_correctness():
    """Test that parity_operator produces mathematically correct results."""
    # Test case 1: Single operator - should give [0, 1] on diagonal
    operators_1 = [FermionOp("c_0")]
    parity_1 = hilbert_space.parity_operator(operators_1, sparse=False)
    expected_1 = np.diag([0, 1])  # |0⟩ has even parity (0), |1⟩ has odd parity (1)
    np.testing.assert_array_equal(
        parity_1, expected_1, err_msg="Single operator parity incorrect"
    )

    # Test case 2: Two operators - should count total number of particles mod 2
    operators_2 = [FermionOp("c_0"), FermionOp("c_1")]
    parity_2 = hilbert_space.parity_operator(operators_2, sparse=False)
    # States: |00⟩, |01⟩, |10⟩, |11⟩
    # Parities: 0,    1,    1,    0  (number of particles mod 2)
    expected_2 = np.diag([0, 1, 1, 0])
    np.testing.assert_array_equal(
        parity_2, expected_2, err_msg="Two operator parity incorrect"
    )

    # Test case 3: Three operators
    operators_3 = [FermionOp("c_0"), FermionOp("c_1"), FermionOp("c_2")]
    parity_3 = hilbert_space.parity_operator(operators_3, sparse=False)
    # States: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
    # Parities:  0,     1,     1,     0,     1,     0,     0,     1
    expected_3 = np.diag([0, 1, 1, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(
        parity_3, expected_3, err_msg="Three operator parity incorrect"
    )


def test_basis_operators_invalid_inputs():
    """Non-operators and missing boson truncation should error."""
    with pytest.raises(ValueError):
        hilbert_space.basis_operators(["not-an-operator"], sparse=False)

    b = BosonOp("b")
    with pytest.raises(ValueError):
        hilbert_space.basis_operators([b], sparse=False)


def test_parity_operator_requires_truncation_for_bosons():
    b = BosonOp("b")
    with pytest.raises(ValueError):
        hilbert_space.parity_operator([b], sparse=False)


def test_parity_operator_mixed_sparse_matches_dense():
    c = FermionOp("c")
    b = BosonOp("b")
    trunc = [3]
    dense = hilbert_space.parity_operator([c, b], sparse=False, truncation=trunc)
    sparse_parity = hilbert_space.parity_operator([c, b], sparse=True, truncation=trunc)
    np.testing.assert_array_equal(dense, sparse_parity.toarray())


@pytest.mark.parametrize("sparse", [False, True])
def test_parity_operator_mixed_occupations(sparse, as_dense):
    c0, c1 = sympy.symbols("c0:2", cls=FermionOp)
    a, b = sympy.symbols("a b", cls=BosonOp)
    parity = hilbert_space.parity_operator(
        [a, c0, b, c1], sparse=sparse, truncation=[2, 1]
    )
    expected = np.diag([sum(state) % 2 for state in np.ndindex(2, 2, 3, 2)])
    np.testing.assert_array_equal(as_dense(parity), expected)


def test_to_matrix_with_precomputed_operator_dict():
    c = FermionOp("c")
    b = BosonOp("b")
    ops = [c, b]
    trunc = 3
    expr = Dagger(c) * b + Dagger(b) * c

    operator_dict = hilbert_space.basis_operators(ops, sparse=False, truncation=trunc)
    mat_default = hilbert_space.to_matrix(expr, ops, sparse=False, truncation=trunc)
    mat_with_dict = hilbert_space.to_matrix(
        expr, ops, sparse=False, truncation=trunc, operator_dict=operator_dict
    )

    assert mat_default.keys() == mat_with_dict.keys()
    for key in mat_default:
        np.testing.assert_allclose(mat_default[key], mat_with_dict[key])
