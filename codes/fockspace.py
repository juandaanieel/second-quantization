from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import sympy
from sympy.core.singleton import S

from functools import reduce
import numpy as np
import scipy.sparse as sp

sigma_0 = np.array([[1, 0],
                    [0, 1]])
sigma_z = np.array([[1,  0],
                    [0, -1]])
sigma_minus = np.array([[0, 1],
                        [0, 0]])

class FockSpace(object):
    def __init__(self, sparse=False):
        self.finalized = False
        self.basis_operators = []
        
        if sparse:
            self.kron = lambda a, b: sp.kron(a, b, format='csr')
            self.eye = lambda N: sp.eye(N, format='csr')
        else:
            self.kron = np.kron
            self.eye = np.eye
        
    def add_fermion(self, name):
        start = len(self.basis_operators)
        
        if isinstance(name, str):
            self.basis_operators.append(FermionOp(name))
            
            return self.basis_operators[start]
        else:
            for n in name:
                self.basis_operators.append(FermionOp(n))

            return self.basis_operators[start:]

    def finalize(self):
        self.matrix_form = {}
        
        N = len(self.basis_operators)
        for i, operator in enumerate(self.basis_operators):
            self.matrix_form[operator] = \
                reduce(self.kron, [ sigma_z ] * i + [ sigma_minus ] +  [self.eye(2**(N-i-1))])
        
        self.finalized = True
    
    def matrix_form_numeric(self, operator):        
        N = 2**len(self.basis_operators)
        dict_matrices = {}
        
        for term, coeff in operator.as_coefficients_dict().items():
            term_mat = float(coeff) * self.eye(N)
            symbol = S.One
            for factor in term.as_ordered_factors():
                if factor in self.matrix_form:
                    term_mat = term_mat @ self.matrix_form[factor]
                elif Dagger(factor) in self.matrix_form:
                    term_mat = term_mat @ self.matrix_form[Dagger(factor)].T.conj()
                else:
                    symbol *= factor
            
            if symbol in dict_matrices.keys():
                dict_matrices[symbol] += term_mat
            else:
                dict_matrices[symbol] = term_mat

        return dict_matrices
