from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.state import Ket
from sympy.core.singleton import S
import sympy

import matplotlib.pyplot as plt
import scipy.linalg as la


from functools import reduce
from collections import defaultdict
import numpy as np
import scipy
import re
from copy import deepcopy
import itertools


sigma_0 = np.array([[1, 0],
                    [0, 1]])
sigma_z = np.array([[1,  0],
                    [0, -1]])
sigma_minus = np.array([[0, 1],
                        [0, 0]])



def basis_operators(
        operators: list[sympy.Expr],
        sparse: bool
        )-> dict[sympy.Expr, np.ndarray | scipy.sparse.csr.csr_matrix:]:
    """
    Transform from second quantized operators to matrix representation in the Fock-space.
    Note that `operators` does not require its elements to be `FermionOp`.
    Note that the fermion ordering follows the ordering in the list.
    """

    if sparse:
        kron = lambda a, b: scipy.sparse.kron(a, b, format='csr')
        eye = lambda N: scipy.sparse.eye(N, format='csr')
    else:
        kron = np.kron
        eye = np.eye

    size = len(operators)
    basis_operators = {}
    for i, operator in enumerate(operators):
        basis_operators[operator] = reduce(kron, [ sigma_z ] * i + [ sigma_minus ] +  [eye(2**(size-i-1))])
    
    return basis_operators

def to_matrix(
        expression: sympy.Expr,
        basis_operators: list[sympy.Expr, np.ndarray | scipy.sparse.csr.csr_matrix]
        )-> dict[sympy.Expr, np.ndarray | scipy.sparse.csr.csr_matrix]:
    """
    Transforms a sympy expression containing fermionic operators to a dictionary the matrix representation of each term.
    """

    example_matrix = next(iter(basis_operators.values()))
    dimension = example_matrix.shape[0]

    if type(example_matrix) == np.ndarray:
        eye = np.eye
    elif type(example_matrix) == scipy.sparse.csr.csr_matrix:
        eye = lambda N: scipy.sparse.eye(N, format='csr')
    else:
        raise ValueError("Unknown matrix type")
    
    dict_matrices = {}
    
    for term, coeff in expression.as_coefficients_dict().items():
        term_mat = float(coeff) * eye(dimension)
        symbol = S.One
        
        for factor in term.as_ordered_factors():
            if factor in basis_operators:
                term_mat = term_mat @ basis_operators[factor]
            elif Dagger(factor) in basis_operators:
                term_mat = term_mat @ basis_operators[Dagger(factor)].T.conj()
            else:
                symbol *= factor
        
        if symbol in dict_matrices.keys():
            dict_matrices[symbol] += term_mat
        else:
            dict_matrices[symbol] = term_mat
    return dict_matrices


def to_operators(
    matrix: np.ndarray | scipy.sparse.csr.csr_matrix,
    basis: list[sympy.Expr],
    )-> sympy.Expr:
    """
    Convert a matrix into a normal ordered operator expression.
    """

    pauli_string = PauliDecomposition(matrix)
    term = S.One

    for string, prefactor in pauli_string.items():
        _term = S.One * prefactor
        _prefactor_exponent = S.One

        for i, key in enumerate(string):
            
            j = i - 1
            if j >= 0:
                _prefactor_exponent *= 2 * Dagger(basis[j]) * basis[j] - 1
            
            if key == 'I':
                _term *= 1 
            elif key == 'X':
                
                _term *= _prefactor_exponent
                _term *= Dagger(basis[i]) + basis[i]
            elif key == 'Y':
                
                _term *= _prefactor_exponent
                _term *= 1j * (Dagger(basis[i]) - basis[i])
            elif key == 'Z':
                _term *= 2 * Dagger(basis[i]) * basis[i] - 1

        term += _term

    term = normal_ordered_form(sy.nsimplify(term).expand(), independent=True)
    return term
            