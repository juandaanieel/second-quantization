# %%
import time
import numpy as np
import sympy
import gc

from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import matplotlib.pyplot as plt

from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian

from second_quantization import fock_space
from second_quantization import hilbert_space
from second_quantization.fock_core.fock_basis import FockBasis

import openfermion
from openfermion.linalg import get_sparse_operator
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
    ops_up = [FermionOp("c_{" + f"{i}" + "\\uparrow}") for i in range(n)]
    ops_dn = [FermionOp("c_{" + f"{i}" + "\\downarrow}") for i in range(n)]
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

    start = time.perf_counter()
    basis = spinful_fermion_basis_1d(L=L)
    time_basis = time.perf_counter() - start

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

# %%

averages = 3
max_n = 7

ns = list(range(2, max_n + 1))

sparse_times_quspin = [0] * len(ns)
sparse_times_sq = [0] * len(ns)
sparse_times_sq_new = [0] * len(ns)

sparse_times_of = [0] * len(ns)

basis_times_sq = [0] * len(ns)
basis_times_sq_new = [0] * len(ns)

basis_times_quspin = [0] * len(ns)
hamiltonian_times_of = [0] * len(ns)


for i, n in enumerate(ns):
    for _ in range(averages):
        H, ops = build_spinful_chain_hamiltonian(n)
        start_basis_sq = time.perf_counter()
        operator_dict = hilbert_space.basis_operators(ops, sparse=True)
        basis_times_sq[i] += time.perf_counter() - start_basis_sq

        start_1 = time.perf_counter()
        hilbert_space.to_matrix(
            H, operators=ops, sparse=True, operator_dict=operator_dict
        )
        sparse_times_sq[i] += time.perf_counter() - start_1

        del H, ops, operator_dict
        gc.collect()
    basis_times_sq[i] /= averages
    sparse_times_sq[i] /= averages
    # =========================
    # Second Quantization
    # =========================
    for _ in range(averages):
        H, ops = build_spinful_chain_hamiltonian(n)
        operator_seq = fock_space.FermionSequence(H, sympy_operators=ops)

        start_basis_sq = time.perf_counter()
        FockBasis(2 * n)
        basis_times_sq_new[i] += time.perf_counter() - start_basis_sq

        start_1 = time.perf_counter()
        operator_seq.compile(data_format="coo")
        sparse_times_sq_new[i] += time.perf_counter() - start_1

        # Clear memory
        del H, ops, operator_seq
        gc.collect()
    basis_times_sq_new[i] /= averages
    sparse_times_sq_new[i] /= averages
    # =========================
    # QuSpin
    # =========================
    for _ in range(averages):
        static, basis, time_basis_qspin = quspin_sparse_hamiltonian(n)
        basis_times_quspin[i] += time_basis_qspin

        start_2 = time.perf_counter()
        H_quspin = hamiltonian(static, [], basis=basis, dtype=np.float64)
        H_sparse = H_quspin.as_sparse_format().static
        sparse_times_quspin[i] += time.perf_counter() - start_2

        # Clear memory
        del static, basis, H_quspin, H_sparse
        gc.collect()
    basis_times_quspin[i] /= averages
    sparse_times_quspin[i] /= averages

    # =========================
    # OpenFermion
    # =========================
    for _ in range(averages):
        start_of = time.perf_counter()
        H = build_spinful_chain_hamiltonian_openfermion(n)
        hamiltonian_times_of[i] += time.perf_counter() - start_of

        H_JW = openfermion.jordan_wigner(H)

        sparse_time_of = time.perf_counter()
        H_sparse_openfermion = get_sparse_operator(H_JW)
        sparse_times_of[i] += time.perf_counter() - sparse_time_of

        # Clear memory
        del H, H_JW, H_sparse_openfermion
        gc.collect()
    hamiltonian_times_of[i] /= averages
    sparse_times_of[i] /= averages
    print(
        f"n={n}: QUSPIN={sparse_times_quspin[i]:.3f}s, "
        f"SQ={sparse_times_sq[i]:.3f}s, "
        f"OF={sparse_times_of[i]:.3f}s"
    )


# %%
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

ns = list(range(2, max_n + 1))
ns = 2 * np.array(ns)
# Panel 1: Matrix build times
axs[0].plot(ns, sparse_times_sq, "o-", label="SQ (to_matrix())")
axs[0].plot(ns, sparse_times_quspin, "o-", label="QuSpin")
axs[0].plot(ns, sparse_times_of, "o-", label="OpenFermion")
axs[0].plot(ns, sparse_times_sq_new, "o-", label="SQ (compile())")

axs[0].set_yscale("log")
# axs[0].loglog()
axs[0].set_xlabel("Number of fermions")
axs[0].set_ylabel("Time (s)")
axs[0].set_title("Sparse Matrix Build Time")
axs[0].legend(frameon=False)

# Panel 2: Basis build times
axs[1].plot(ns, np.array(basis_times_sq), "o-", label="SQ")
axs[1].plot(ns, np.array(basis_times_quspin), "o-", label="QuSpin")
axs[1].plot(ns, np.array(sparse_times_of), "o-", label="OpenFermion")
axs[1].plot(ns, np.array(basis_times_sq_new), "o-", label="SQ (compile())")

axs[1].set_yscale("log")
# axs[1].loglog()
axs[1].set_xlabel("Number of fermions")
axs[1].set_ylabel("Time (s)")
axs[1].set_title("Basis Build Time")
axs[1].legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
