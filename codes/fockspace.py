from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import sympy
from sympy.core.singleton import S
from sympy.physics.quantum.operatorordering import normal_ordered_form

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
        """
        Generate the basis of fermionic operators from a name list
        """
        start = len(self.basis_operators)
        
        if isinstance(name, str):
            self.basis_operators.append(FermionOp(name))
            
            return self.basis_operators[start]
        else:
            for n in name:
                self.basis_operators.append(FermionOp(n))

            return self.basis_operators[start:]

    def even_odd_ordering(self):
        """
        Sort the elements of the parity operator as even and odd
        """
        # parity operator
        ops = np.prod([1-2*Dagger(op)*op for op in self.basis_operators])
        ops = normal_ordered_form(ops.expand(), independent=True)
        # compute matrix representation
        terms = self.matrix_form_numeric(ops)
        parity_order = terms[S.One].diagonal()
        # sort and return elements
        even = np.where(parity_order>0)[0]
        odd = np.where(parity_order<0)[0]
        self.permutation = np.array([even, odd]).flatten()
        

    def finalize(self):
        self.matrix_form = {}
        # compute matrix representation of cs and c daggers
        N = len(self.basis_operators)
        for i, operator in enumerate(self.basis_operators):
            self.matrix_form[operator] = \
                reduce(self.kron, [ sigma_z ] * i + [ sigma_minus ] +  [self.eye(2**(N-i-1))])

        # compute order of even and odd subspaces
        self.even_odd_ordering()
        self.finalized = True
    
    def matrix_form_numeric(self, operator, parity_sectors=False):        
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
                
        # after finalized it sorts the parity states
        if self.finalized:
            for key, value in dict_matrices.items():
                value[:, :] = value[:, self.permutation]
                value[:, :] = value[self.permutation, :]
                
                if value[N//2:, :N//2].all() != 0:
                    raise("Incorrect ordering of parity sectors")

        # return separate matrices for even odd sectors
        if parity_sectors:
            for key, value in dict_matrices.items():
                sectors = np.array([value[:N//2, :N//2], value[N//2:, N//2:]])
                dict_matrices[key] = sectors

        return dict_matrices
