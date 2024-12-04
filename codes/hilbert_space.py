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
import scipy.sparse as sp
import re
from copy import deepcopy
import itertools


sigma_0 = np.array([[1, 0],
                    [0, 1]])
sigma_z = np.array([[1,  0],
                    [0, -1]])
sigma_minus = np.array([[0, 1],
                        [0, 0]])

SPINLESS = 0
SPIN_UP = 1
SPIN_DOWN = 2

class HilbertSpace():
    """
    class for handling of representations in a Hilbert space
    """
    def __init__(self, sparse=False):
        self.finalized = False
        self.dimension = None
        self.basis_operators = {} #strictly the single annihilators and how they were submitted on creation
        self.basis_spin = []
        self.op_matrices = None
        self.parity_op = None
        self.all_operators = None
        
        if sparse:
            self.kron = lambda a, b: sp.kron(a, b, format='csr')
            self.eye = lambda N: sp.eye(N, format='csr')
            self.solve = lambda A,b: sp.linalg.spsolve(A, b)
            self.flatten = lambda A: A.reshape(1,int(A.shape[0]**2))
            self.stack = lambda l: sp.vstack(l).tocsr().T
        else:
            self.kron = np.kron
            self.eye = np.eye
            self.solve = lambda A,b: la.solve(A, b)
            self.flatten = lambda A: A.flatten()
            self.stack = lambda l: np.array(l).T

    def add_fermion(self, name, spinful=False):
        """
        Add a fermionic degree of freedom to the Hilbert space

        Parameters:
        name:       String or list of Strings
                    name(s) of the fermionic d.o.f.s that are submitted to
                    the Hilbert space.
                    
        spinful:    Boolean
                    Boolean value determining wether the Fermions that are
                    created from the `name` parameter are spinful or not.
                    If 

        Return:
        """
        start = len(self.basis_operators)
        
 
        def _add_spin_to_name(name):
            if ((m := re.match(r"(.+)_\{(.+)\}$", name)) or
                (m := re.match(r"(.+)_([A-Za-z0-9])$", name)) or
                (m := re.match(r"(.+)_(\\\S+)$", name)) ):
                return (m[1] + "_{" + m[2] + r",\uparrow}",
                        m[1] + "_{" + m[2] + r",\downarrow}")
            else:
                raise ValueError("Cannot add spin to Latex expression '" 
                                 + name + "'")
        
        def _add_single_fermion(name, spinful):
            if spinful:
                name_up, name_down = _add_spin_to_name(name)
                fermions = [FermionOp(name_up), FermionOp(name_down)]
                extend = {fermion: None for fermion in fermions}
                self.basis_operators = deepcopy(self.basis_operators)|extend
                self.basis_spin.extend([SPIN_UP, SPIN_DOWN])
                return fermions
                                       
            else:
                fermion = [FermionOp(name)]
                self.basis_operators[FermionOp(name)] = None
                self.basis_spin.append(SPINLESS)
                return fermion

        
        if isinstance(name, str):
            fermions = _add_single_fermion(name, spinful)
            
            return fermions
        else:
            fermions = []
            for n in name:
                fermions += _add_single_fermion(n, spinful)

            return fermions

    # Vectors
    def symbolic_vectors(self):
        """
        Generator function for the Fock-states that span the Hilbert space
        """
        
        # This is not optimized at all, but I just want to get something to work
        N = 2**len(self.basis_operators)
        
        self.symbolic_vecs = [None] * N
        
        vacuum_vec = np.zeros(shape=(N,))
        vacuum_vec[0] = 1
        
        sympy_one = sympy.sympify(1)
        
        for i in range(N):
            reverse_bitstring = format(i, 'b')[::-1]
            reverse_bitstring += "0" * (len(self.basis_operators) - len(reverse_bitstring))
            
            # find the index corresponding to the string of operators 
            # defined by `reverse_bitstring`. Actually, it should be 
            # possible to skip this.
            expr = sympy_one
            for index, c in enumerate(reverse_bitstring):
                if c == '1':
                    expr = expr * Dagger(self.basis_operators[index])
            
            vec = self.matrix_numeric(expr) @ vacuum_vec
            ket_index = vec.nonzero()[0][0]
            
            # now make the ket symbol for the vector:
            ket_spec = []
            for index, c in enumerate(reverse_bitstring):
                spin = self.basis_spin[index]
                
                if spin == SPIN_UP:
                    if c == '1': ket_spec.append(r"\uparrow")
                    else: ket_spec.append(r"0")
                elif spin == SPIN_DOWN:
                    if c == '1':
                        if ket_spec[-1] == r"0": ket_spec[-1] = r"\downarrow"
                        else: ket_spec[-1] += r"\downarrow"
                    # note that if down spin is unoccupied, the spin up formatting 
                    # already gives the right string
                else:
                    if c == '1': ket_spec.append("1")
                    else: ket_spec.append("0")
        
            self.symbolic_vecs[ket_index] = Ket(",".join(ket_spec))

    
    def vector_symbolic(self, vec, zero_tol=1e-10):
        """
        Bring a specific representation back to an abstract 
        state Fock-space representation.

        Parameters:
        vec: np.ndarray
                Representation of the state in a basis

        zero_tol: float
                Tolerance of the rounding

        Returns:
        expr: sympy.expr
                Abstract state representation
        """
        
        expr = 0
        
        for index, value in enumerate(vec):
            if abs(value) > zero_tol:
                expr += value * self.symbolic_vecs[index]
                
        return expr

    #Operators
    def finalize(self):
        """
        Finalize HilbertSpace creation by casting the individual CAO to matrices
        """
        # compute matrix representation of cs and c daggers
        N = len([operator for operator in self.basis_operators if type(operator) == FermionOp])
        for i, kv in enumerate(self.basis_operators.items()):
            if (kv[1] is None and 
                type(kv[0]) == FermionOp):
                self.basis_operators[kv[0]] = reduce(self.kron, [ sigma_z ] * i + [ sigma_minus ] +  [self.eye(2**(N-i-1))])
        self.finalized = True
        self.dimension = 2**N


    def get_combinations(self, subset):
        """
        Get all combinations of annihilation operators from a list of symbolic FermionOps

        Parameters:
        subset: list of FermionOps
                list of Fermionic operators that are contained in the Fock-space
        """

        N_max = 2**len(subset)
        n_ops = len(subset)
        operator_list = {}
        symbolic_list = []

        for i in range(N_max+1):
            bit_string = [0 for _ in range(n_ops-len(list(map(int, format(i, 'b')))))] + list(map(int, format(i, 'b')))
            symbolic_list.append(np.prod([b_op**bit for b_op, bit in zip(subset, bit_string)]))

        for symbol in symbolic_list:
            operator_list[symbol] = self.matrix_numeric(symbol)[1]
            
        return operator_list
        

    def matrix_numeric(self, operator):  
        """
        Generate a matrix representation in Fock-space of an operator

        Parameters:
        operator: sympy expression
                    Symbolic expression of the operator to cast to the Fock-space
        """
        N = self.dimension
        dict_matrices = {}
        
        for term, coeff in operator.as_coefficients_dict().items():
            term_mat = float(coeff) * self.eye(N)
            symbol = S.One
            #print(term)
            for factor in term.as_ordered_factors():
                if factor in self.basis_operators:
                    term_mat = term_mat @ self.basis_operators[factor]
                elif Dagger(factor) in self.basis_operators:
                    term_mat = term_mat @ self.basis_operators[Dagger(factor)].T.conj()
                else:
                    symbol *= factor
            
            if symbol in dict_matrices.keys():
                dict_matrices[symbol] += term_mat
            else:
                dict_matrices[symbol] = term_mat
        return dict_matrices

    def get_operator_decomposition(self, operator, get_basis=False):
        '''
        Expand a given operator in terms of basis CAOs.

        Parameters:
        operator: np.ndarray
                Matrix representation of an operator in the Fock-basis

        get_basis: boolean
                Boolean deciding whether the dict of operators spanning
                the Hilbert space should also be returned

        Returns:
        vector: np.ndarray
                vector of coefficients representing the operators for 
                all the set of normal ordered operators that span the
                Fock-space
        '''
        if type(operator) == dict:
            return sympy.nsimplify(sum([k*self.get_operator_decomposition(v) for k,v in operator.items()]))
        if self.all_operators == None:
            basis_operators = ([Dagger(op) for op in reversed(self.basis_operators.keys())] +
                list(self.basis_operators.keys()))
            self.all_operators = self.get_combinations(basis_operators)
            
        op_list = list(self.all_operators.keys())
        mat_list = list(self.all_operators.values())
        flat_mat_list = [self.flatten(v) for v in mat_list]
        flat_mat_matrix = self.stack(flat_mat_list)

        if type(flat_mat_matrix) == np.ndarray:
            assert np.allclose(la.det(flat_mat_matrix), 1)
        assert np.allclose(self.all_operators[1].shape, operator.shape)

        solution = self.solve(flat_mat_matrix, self.flatten(operator).T)

        return sum([s*op for s,op in zip(solution, list(self.all_operators.keys()))])

    # Convenience operators
    def get_parity_op(self):
        ops = np.prod([(1-2*Dagger(op)*op) for op in self.basis_operators])
        return self.matrix_numeric(ops.expand())[1]
    
    def partial_trace_generators(self, subset, complement):
        """
        Generate the projectors required to compute the partial trace of a subset of operators
        """
        subset_operators = self.get_combinations(subset)
        complement_operators = self.get_combinations(complement)

        size = list(subset_operators.values())[0].shape[0]
        vacuum = np.zeros(size)
        vacuum[0] = 1

        vecs = []

        for element in complement_operators.values():
            aux = []
            for g in subset_operators.values():
                generator = g @ element
                aux.append(vacuum @ generator)
            vecs.append(aux)

        return np.array(vecs)
        
        




