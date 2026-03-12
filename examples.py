import numpy as np
import sympy as sp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger


from second_quantization import fock_space
from second_quantization.fock_space import c_down, c_up, a_down, a_up


# # Example of FermionSequence()


def build_H_fermi_hubbard(N):
    ## Make ECT terms
    ECT = (c_down * (~c_down >> 1) + c_up * (~c_up >> 1)) >> np.arange(1, N - 1)

    MU = (c_down * (~c_down) + c_up * (~c_up)) >> np.arange(1, N)

    U = (c_up * c_down * a_up * a_down) >> np.arange(1, N)

    return U, MU, ECT


N = 2
U, MU, ECT = build_H_fermi_hubbard(N)
H = U + MU + ECT
H

sparse_data = H.to_sparse(N)
coo_sparse = sparse_data.to_sparse_coo()
csr_sparse = sparse_data.to_sparse_csr()
np_array = sparse_data.to_array()

"""
Updating values can be done in multiple ways. Most convenient is via subsequence
"""
H[U] = 10
H[ECT] = 2
H[MU[0]] = 5
H[MU[1]] = 20
H

H.to_sparse(2)

"""
Currently works with sympy symbols as weights but not yet in the same structure as hilbert_space
ToDo - implement the same FermionOp instantiation
Also - fix printing in a nicer way (or convert to SymPy before printing)
"""
u, t, mu = sp.symbols("U, t,mu")
H_symb = u * U + mu * MU + t * ECT
H_symb


# # From Sympy to FermionSequence


def build_spinful_chain_hamiltonian(n):
    """
    Builds a chain Hamiltonian for spinful fermions with:
    - nearest-neighbor hopping for both spins,
    - onsite Coulomb interaction between up/down spins,
    - chemical potential for both spins.
    There are 2*n fermionic modes: c_{i,up}, c_{i,down} for i in 0..n-1.
    """
    t, U, mu = sp.symbols("t U mu", real=True)
    ops_up = [FermionOp("c_{" + f"{i}" + "\\uparrow}") for i in range(n)]
    ops_dn = [FermionOp("c_{" + f"{i}" + "\\downarrow}") for i in range(n)]
    opers_sorted = []
    for idx in range(len(ops_up)):
        opers_sorted.append(ops_dn[idx])
        opers_sorted.append(ops_up[idx])

    hopping = 0
    interaction = 0
    chemical = 0

    for i in range(n - 1):
        # Hopping for up
        hopping += -t * (
            Dagger(ops_up[i]) * ops_up[i + 1] + Dagger(ops_up[i + 1]) * ops_up[i]
        )
        # Hopping for down
        hopping += -t * (
            Dagger(ops_dn[i]) * ops_dn[i + 1] + Dagger(ops_dn[i + 1]) * ops_dn[i]
        )

    for i in range(n):
        # Onsite Coulomb interaction
        interaction += (
            U * (Dagger(ops_up[i]) * ops_up[i]) * (Dagger(ops_dn[i]) * ops_dn[i])
        )
        # Chemical potential for both spins
        chemical += -mu * (
            Dagger(ops_up[i]) * ops_up[i] + Dagger(ops_dn[i]) * ops_dn[i]
        )

    H = hopping + interaction + chemical
    return H, opers_sorted


N = 2
## Built Sympy hamiltonian and FermionOps
H_sympy, fermops = build_spinful_chain_hamiltonian(N)
H_sympy

## Convert from Sympy to FermionSequence
as_sequence = fock_space.to_sequence(H_sympy, fermops)
as_sequence

## Build sparse representation (will initialize an empty array for vals until values are determined)
## TODO - check if its better to instead allocate an array of 0s? not sure if the garbage of empty can cause problems.
as_sequence.to_sparse(N)
