import numpy as np
import scipy
import pytest
import sympy
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
