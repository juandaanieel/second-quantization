import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from sympy.core.singleton import S
from functools import reduce
import numpy as np

tau_z = np.array([[1, 0], [0, -1]])
tau_x = np.array([[0, 1], [1, 0]])
tau_y = 1j * np.array([[0, -1], [1, 0]])


def jw_string(pm, index, n_total):
    """
    Build matrix representation of second quantized operators
    TODO: this function is very ugly, rewrite it
    """
    for i in range(n_total):
        # first two ifs are for initialization
        if i == 0 and i != index:
            aux = tau_z
        elif i == 0 and i == index:
            aux = 1/2*(tau_x - pm * 1j*tau_y)
        # remaining ifs are for building the jw string
        elif i < index:
            aux = np.kron(aux, tau_z)
        elif i == index:
            aux = np.kron(aux, 1/2*(tau_x - pm * 1j*tau_y))
        else:
            aux = np.kron(aux, np.eye(2))
    return aux



def many_body_matrix(symbolic_ham, basis):
    """
    Generates a dictionary with keys as symypy terms of the fermionic Hamiltonian
    
    Parameters:
        symbolic_ham: Normal ordered sympy.Fermion's
        basis: tuple with all annihilations ops used
    Returns:
        dict
    """
    n_total = len(basis)
    basis_dagger = [Dagger(b) for b in basis]
    
    dict_expr = symbolic_ham.as_coefficients_dict()
    dict_matrices = dict()
    
    # iterate over all terms in the Hamiltonian (normal ordered)
    for term, n_coeff in dict_expr.items():
        # initialize symbolic coefficient and matrix
        sym_coef = S.One
        matrix = np.eye(2**n_total, dtype=int)
        
        # iterate over all symbols of the expresion
        # handle constant terms
        if len(term.args) == 0:
            sym_coef = term

        # terms with c and c daggers
        else:
            for factor in term.args:
                if type(factor) is FermionOp:
                    try:
                        index = basis.index(factor)
                        pm = 1
                    except ValueError:
                        index = basis_dagger.index(factor)
                        pm = -1
                    matrix = jw_string(pm=pm, index=index, n_total=n_total) @ matrix
                else:
                    sym_coef *= factor

        # add each term to dictionary
        if sym_coef in dict_matrices.keys():
            dict_matrices[sym_coef] += matrix
        else:
            dict_matrices[sym_coef] = matrix

    return dict_matrices