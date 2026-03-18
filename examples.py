import numpy as np
import sympy as sp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from second_quantization import fock_space
from second_quantization.fock_space import c_down, c_up

# %load_ext autoreload
# %autoreload 2

# # Sympy to FermionSequence


"""
fock_space.to_sequence() takes a sympy expression and a list of FermionOps setting the normal order, and parses it to
the underlying FermionSequence() while maintaining the relation to the FermionOps.
"""

t, U, mu = sp.symbols("t U mu", real=True)


def build_spinful_chain_hamiltonian(n):
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
    return H, ops_up + ops_dn


N = 9
## Example: Built Sympy hamiltonian and FermionOps
H_sympy, ops = build_spinful_chain_hamiltonian(N)
H_sympy

H_seq = fock_space.to_sequence(H_sympy, ops)
H_seq  ## This H is now a FermionSequence()

"""
To create the data representation
"""
H_data = H_seq.compile(data_format="coo")
H_data.get()  ## Obtain the array of data

"""
To fill the sparse array, call H_data with a dictionary mapping the sympy symbols to values
"""
H_data({t: 1.0, U: 2.5, mu: 3})  ## Fill the parametrized values
H_data.update_values({t: 1.0, U: 2.5, mu: 3})  ## Aliased alternative for readability

"""
Any FermionSequence() can be converted back to sympy
"""
sympy_expr, operators = H_seq.to_sympy()
sympy_expr  ## This is now a sympy expression

# ## Compact example

H_sympy, ops = build_spinful_chain_hamiltonian(n=3)  ## Built sympy array
H = fock_space.to_sequence(H_sympy, ops)  ## Map Sympy to FermionSequence
H_data = H.compile(
    data_format="coo"
)  ## Compile in default fock basis, returning COO format
H_data({t: 5.0, U: 2.5, mu: 3})  ## Set parametrized values
H_data.get()  ## Get data

# # Example usage of FermionSequence()


"""
Implementation via FermionSequence() is fully self-contained in terms of implementing fermionic algebra without sympy
and can be used to create and update Hamiltonians by itself.
Multiple ways to initialize a sequenec are implemented, fastest is just to start with operators c_up/c_down and combine them to make terms.
"""


def build_H_fermi_hubbard(N):
    ## The bitshift (>>) operators shifts an operator to act on a different site
    ## The invert (~) operator takes the conjugate
    ECT = (c_down * (~c_down >> 1) + c_up * (~c_up >> 1)) >> np.arange(1, N - 1)
    ECT += ~ECT

    ## Same operations are also implemented via dag() and extend()
    MU = (c_down * (c_down.dag()) + c_up * (c_up.dag())).extend(np.arange(1, N))

    ## Any sequence of operators can be made just by adding and multiplying terms
    U = (c_up * c_down * ~c_up * ~c_down) >> np.arange(1, N)

    ## It is convenient to separate terms of interest for easily updating their weights later
    return U, MU, ECT


N = 2
U, MU, ECT = build_H_fermi_hubbard(N)
H = U + MU + ECT
H

"""
Initializing values can be done in multiple ways. Most convenient is by setting the values via subsequences:
"""
H[U] = -10
H[ECT] = 2
H[MU[0]] = 5
H[MU[1]] = 20
H

"""
Weights can also be parametrized via Sympy expressions, which will need to be given a value later.
@TODO currently this will as of now cause different behavior compared to initializing a sequence below from a Sympy expression, need to make this consistent.
"""
u, t, mu = sp.symbols("U, t,mu")
H_symbolic_weights = u * U + mu * MU + t * ECT
H_symbolic_weights

"""
Once Hamiltonian is set, compile() creates a representation of the Hamiltonian matrix in chosen data_format, storing additional info for quickly updating values later.
"""
U, MU, ECT = build_H_fermi_hubbard(N)
H = U + MU + ECT
H_data = H.compile(data_format="coo")
"""
From here one can obtain the data array for doing things
"""
H_data.get()  ## Returns the native data-type of whichever data backend was chosen (so that by default it returns the data-type the structure was optimized for)

# # Potential benefit: faster fermionic operations

"""
The FermionSequence() does the fermionic algebra by extending and sorting lists of integers, which seems faster then relying on sympy's own functions
As an example below code creates a complex H and operator Q that are known to commute and compares speed of calculating the commutator.
Currently FermionSequence() uses python loops for algebra operations, a x10 speed up is achieved by handling that e.g. in cython.
"""

# +
"""
Bunch of functions to set up the system
"""
omega = sp.nsimplify(sp.exp(2 * sp.pi * sp.I / 3))
a_up = c_up.dag()
a_down = c_down.dag()
n_up = c_up * a_up
n_down = c_down * a_down
P = -c_down * a_down * c_up * a_up + 1

## U and V matrices
U = sp.Matrix(3, 3, lambda i, j: 0)
U[0, 1] = 1
U[1, 2] = 1
U[2, 0] = 1

V = sp.Matrix(3, 3, lambda i, j: 0)
V[0, 0] = 1
V[1, 1] = omega
V[2, 2] = omega**2

base_vec = [
    [P * (-n_up - n_down + 1) * P, P * a_down * P, P * a_up * P],
    [P * c_down * P, P * n_down * P, P * c_down * a_up * P],
    [P * c_up * P, P * c_up * a_down * P, P * n_up * P],
]


def matrix_to_operator(matrix):
    oper = fock_space.FermionSequence()
    for i in range(3):
        for j in range(3):
            oper += matrix[i, j] * base_vec[i][j]
    oper.remove_zero_weight()
    return oper


def odd_even_split(oper):
    oper_list_even = []
    oper_list_odd = []
    weight_list_even = []
    weight_list_odd = []
    for idx, w in enumerate(oper.oper_list):
        if len(w) % 2:
            oper_list_odd.append(w)
            weight_list_odd.append(oper.weights[idx])
        else:
            oper_list_even.append(w)
            weight_list_even.append(oper.weights[idx])

    even_op = fock_space.FermionSequence(
        oper_list_even, weights=weight_list_even, bypass_parse=True
    )
    odd_op = fock_space.FermionSequence(
        oper_list_odd, weights=weight_list_odd, bypass_parse=True
    )
    return even_op, odd_op


def make_operators(W):
    U_prime = sp.simplify(W.H * U * W)
    V_prime = sp.simplify(W.H * V * W)
    u = matrix_to_operator(U_prime)
    v = matrix_to_operator(V_prime)

    for idx, w in enumerate(u.weights):
        u.weights[idx] = sp.simplify(sp.expand(w))
    u.remove_zero_weight()

    for idx, w in enumerate(v.weights):
        v.weights[idx] = sp.simplify(sp.expand(w))
    v.remove_zero_weight()

    prod = u * v
    for idx, w in enumerate(prod.weights):
        prod.weights[idx] = sp.simplify(sp.expand(w))
    prod.remove_zero_weight()

    return u, v


(
    sigma,
    tau,
) = make_operators(sp.diag(1, 1, 1))

# +
"""
Creating system with known structure
"""
Proj = (-2 * P * c_up * P * P * a_up * P + 1) * (-2 * P * c_down * P * a_down * P + 1)
even, odd = odd_even_split(sigma)
sigma_graded = (even >> 1) + Proj * (odd >> 1)

sigma1 = sigma
sigma2 = sigma_graded
tau1 = tau
tau2 = tau >> 1

chi1 = sigma1
chi2 = omega * chi1 * tau1
chi3 = tau1 * sigma2
chi4 = omega * chi3 * tau2
# -

"""
Building H
"""
H = np.conj(omega) * (~chi2) * chi3 + omega * (~chi3 * chi2)
H.simplify()
H.sort()
H

"""
Building Q
"""
Q = tau1 * tau2
Q

"""
Getting the commutator with the FermionSequence() structure
"""
# %time commutator = Q*H- H*Q
commutator = Q * H - H * Q
commutator.simplify()
commutator
