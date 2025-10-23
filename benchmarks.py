# %%
import time
import numpy as np
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import matplotlib.pyplot as plt

from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian

from second_quantization import hilbert_space

import openfermion
from openfermion.transforms import get_sparse_operator
from openfermion.ops import FermionOperator


# %%
def build_chain_hamiltonian(n):
    """Builds a chain Hamiltonian with nearest-neighbor hopping and density-density interaction."""
    ops = [FermionOp(f"c_{i}") for i in range(n)]
    hopping = 0
    interaction = 0
    t, U = sympy.symbols("t U", real=True)
    for i in range(n - 1):
        hopping += -t * (Dagger(ops[i]) * ops[i + 1] + Dagger(ops[i + 1]) * ops[i])
        interaction += U * (Dagger(ops[i]) * ops[i]) * (Dagger(ops[i + 1]) * ops[i + 1])
    return hopping + interaction, ops


def build_spinful_chain_hamiltonian(n):
    """
    Builds a chain Hamiltonian for spinful fermions with:
    - nearest-neighbor hopping for both spins,
    - onsite Coulomb interaction between up/down spins,
    - chemical potential for both spins.
    There are 2*n fermionic modes: c_{i,up}, c_{i,down} for i in 0..n-1.
    """
    t, U, mu = sympy.symbols("t U mu", real=True)
    ops_up = [FermionOp(f"c_{i}_up") for i in range(n)]
    ops_dn = [FermionOp(f"c_{i}_dn") for i in range(n)]
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
    ops = ops_up + ops_dn
    return H, ops


# %%
def build_spinful_chain_hamiltonian_openfermion(n, t=1.0, U=1.0, mu=0.0):
    """
    Builds a chain Hamiltonian for spinful fermions using OpenFermion.
    - n: number of sites
    - t: hopping amplitude
    - U: onsite interaction
    - mu: chemical potential
    Returns: FermionOperator
    """
    H = FermionOperator()
    for i in range(n - 1):
        # Hopping for up
        H += -t * (
            FermionOperator(f"{2*i}^ {2*(i+1)}") + FermionOperator(f"{2*(i+1)}^ {2*i}")
        )
        # Hopping for down
        H += -t * (
            FermionOperator(f"{2*i+1}^ {2*(i+1)+1}")
            + FermionOperator(f"{2*(i+1)+1}^ {2*i+1}")
        )
    for i in range(n):
        # Onsite Coulomb interaction
        H += U * FermionOperator(f"{2*i}^ {2*i} {2*i+1}^ {2*i+1}")
        # Chemical potential for both spins
        H += -mu * (
            FermionOperator(f"{2*i}^ {2*i}") + FermionOperator(f"{2*i+1}^ {2*i+1}")
        )
    return H


# %%
def quspin_sparse_hamiltonian(L):
    """Builds a sparse Hamiltonian for a 1D spinful fermion model."""
    J = 1.0  # Hopping matrix element
    U = 2.0  # Onsite interaction strength
    mu = 0.5  # Chemical potential

    start = time.time()
    basis = spinful_fermion_basis_1d(L=L)
    time_basis = time.time() - start

    # Only define one direction for hopping; QuSpin adds the Hermitian conjugate
    hop_right = [[-J, i, (i + 1) % L] for i in range(L)]  # hopping to the right PBC
    hop_left = [[J, i, (i + 1) % L] for i in range(L)]  # hopping to the left PBC
    potential = [[-mu, i] for i in range(L)]
    interaction = [[U, i, i] for i in range(L)]

    static = [
        ["+-|", hop_left],  # up hop left
        ["-+|", hop_right],  # up hop right
        ["|+-", hop_left],  # down hop left
        ["|-+", hop_right],  # down hop right
        ["n|", potential],  # Onsite potential for spin up
        ["|n", potential],  # Onsite potential for spin down
        ["n|n", interaction],  # Spin up-spin down interaction
    ]
    return static, basis, time_basis


# %%
max_n = 11

sparse_times_quspin = []
sparse_times_sq = []
sparse_times_of = []

basis_times_sq = []
basis_times_quspin = []
hamiltonian_times_of = []

ns = list(range(2, max_n + 1))

for i, n in enumerate(ns):
    # Second quantization
    H, ops = build_spinful_chain_hamiltonian(n)

    start_basis_sq = time.time()
    operator_dict = hilbert_space.basis_operators(ops, sparse=True)
    basis_times_sq.append(time.time() - start_basis_sq)

    start_1 = time.time()
    hilbert_space.to_matrix(H, operators=ops, sparse=True, operator_dict=operator_dict)
    sparse_times_sq.append(time.time() - start_1)

    # QuSpin
    static, basis, time_basis_qspin = quspin_sparse_hamiltonian(n)
    basis_times_quspin.append(time_basis_qspin)

    start_2 = time.time()
    H_quspin = hamiltonian(static, [], basis=basis, dtype=np.float64)
    H_sparse = H_quspin.as_sparse_format().static
    sparse_times_quspin.append(time.time() - start_2)

    # OpenFermion
    # Same result as
    # H = openfermion.fermi_hubbard(n, 1, 1, 1, periodic=False)
    # H_sparse = get_sparse_operator(H)
    start_of = time.time()
    H = build_spinful_chain_hamiltonian_openfermion(n)
    hamiltonian_times_of.append(time.time() - start_of)
    H_JW = openfermion.jordan_wigner(H)
    sparse_time_of = time.time()
    H_sparse_openfermion = get_sparse_operator(H_JW)
    sparse_times_of.append(time.time() - sparse_time_of)

    print(
        f"n={n}: QUSPIN={sparse_times_quspin[-1]:.3f}s, SQ={sparse_times_sq[-1]:.3f}s, OF={sparse_times_of[-1]:.3f}s"
    )


# %%
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

ns = list(range(2, max_n + 1))
ns = 2 * np.array(ns)
# Panel 1: Matrix build times
axs[0].plot(ns, sparse_times_sq, "o-", label="SQ (to_matrix)")
axs[0].plot(ns, sparse_times_quspin, "o-", label="QuSpin")
axs[0].plot(ns, sparse_times_of, "o-", label="OpenFermion")
axs[0].set_yscale("log")
# axs[0].loglog()
axs[0].set_xlabel("Number of fermions")
axs[0].set_ylabel("Time (s)")
axs[0].set_title("Sparse Matrix Build Time")
axs[0].legend(frameon=False)

# Panel 2: Basis build times
axs[1].plot(ns, basis_times_sq, "o-", label="SQ")
axs[1].plot(ns, basis_times_quspin, "o-", label="QuSpin")
# axs[1].plot(ns, hamiltonian_times_of, 'o-', label='OpenFermion')
axs[1].set_yscale("log")
# axs[1].loglog()
axs[1].set_xlabel("Number of fermions")
axs[1].set_ylabel("Time (s)")
axs[1].set_title("Basis Build Time")

plt.tight_layout()
plt.show()

# %%
