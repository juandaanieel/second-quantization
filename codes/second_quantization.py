from itertools import product, combinations
from functools import partial

import numpy as np
import sympy
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum import Dagger
from multiprocessing import Pool

# this work only on the latest version of sympy on github.
# Otherwise, its gonna be incorrect, because pypy/conda-forge current version is buggy


def collect_constant(expr):
    """Collect constant terms in a fermionic expression.

    Parameters
    ==========

    expr : sympy expression

    Returns

    =======

    constant_terms : sympy expression
    """
    expr = normal_ordered_form(expr.expand(), independent=True, recursive_limit=100)
    return sum(
        [
            term
            for term in expr.as_ordered_terms()
            if not term.has(sympy.physics.quantum.Operator)
        ]
    )


def fermionic_basis_states(operators):
    """Generate a basis of states from a list of fermionic operators.

    Parameters
    ==========

    operators : list of sympy FermionOp objects

    Returns

    =======

    states : list of sympy FermionOp objects
        Each element of the list is a state in the many-body basis
    """

    basis = []
    n_ops = len(operators)
    for n in range(n_ops):
        combos = combinations(operators, n + 1)
        basis.extend(combos)
    states = []
    states.extend([sympy.Mul(*ops) for ops in basis])
    return states


def braket_product(braket, ham):
    return collect_constant(braket[0] * ham * Dagger(braket[1]))


def matrix_elements(ham, basis, num_processes=5):
    """Generate the matrix elements of a fermionic operator in a given basis.

    Parameters
    ==========

    ham : sympy FermionOp object
        The operator whose matrix elements are to be calculated

    basis : list of sympy FermionOp expressions
        The basis in which the matrix elements are to be calculated


    num_processes : int
        Number of processes to use for multiprocessing (default=5)

    Returns

    =======

    matrix_elements : list of sympy expressions
        The matrix elements of the operator in the given basis
    """
    all_brakets = product(basis, basis)

    matrix_elements = []
    with Pool(num_processes) as mp_pool:
        matrix_elements.extend(mp_pool.map(partial(braket_product, ham=ham), all_brakets))
    return matrix_elements


def to_matrix(H):
    """Compute a matrix representation of a sympy expression with fermion operators."""
    # Add an identity operator to all symbols so that we always work with operators
    H = H.subs({
        s: sympy.physics.quantum.IdentityOperator() * s for s in H.free_symbols
        if not isinstance(s, sympy.physics.quantum.Operator)
    })
    # Choose an order of Fermionic operators (or should we ask for it?
    fermions = [
        s for s in H.free_symbols
        if (
            isinstance(s, sympy.physics.quantum.fermion.FermionOp)
            and s.is_annihilation
        )
    ]
    # Compute matrix representations
    s_minus = sympy.Matrix([[0, 1], [0, 0]])
    s_z = sympy.Matrix([[1, 0], [0, -1]])
    s_0 = sympy.eye(2)
    matrix_subs = {
        op: sympy.kronecker_product(*(i * [s_z] + [s_minus] + (len(fermions) - i - 1) * [s_0]))
        for i, op in enumerate(fermions)
    }
    matrix_subs.update({Dagger(op): Dagger(mat) for op, mat in matrix_subs.items()})
    matrix_subs[sympy.physics.quantum.IdentityOperator()] = sympy.eye(2**len(fermions))

    # Generate basis
    basis = [(sympy.S.One,)]
    for n in range(len(fermions)):
        basis.extend(list(combinations(fermions, n + 1)))
    reversed_basis = list(reversed(basis))
    reversed_basis[-1] = (sympy.S.Zero,)

    basis_matrices = []
    for b, nb in zip(basis, reversed_basis):
        expr = [Dagger(op) * op for op in b]
        expr.extend([sympy.physics.quantum.IdentityOperator()-Dagger(op) * op for op in nb])
        basis_matrices.append(sympy.Mul(*expr).expand())
    basis_matrices = [b.subs(matrix_subs, simultaneous=True).expand() for b in basis_matrices]
    basis_order = [np.nonzero(np.array(b.diagonal(), dtype=int)[0])[0][0] for b in basis_matrices]
    effective_basis = [sympy.Mul(*basis[i]) for i in np.argsort(basis_order)]

    return H.subs(matrix_subs, simultaneous=True).expand(), effective_basis