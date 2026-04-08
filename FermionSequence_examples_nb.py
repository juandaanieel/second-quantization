import numpy as np
import matplotlib.pyplot as plt
from second_quantization import fock_space
from second_quantization.fock_core.fock_basis import ParityMask, DoubleOccupancyMask

from example_models import build_spinful_chain_hamiltonian

""
# %load_ext autoreload
# %autoreload 2

"""
# Sympy to FermionSequence
The fock_space.FermionSequence() class accepts a sympy expression and a list of FermionOps setting the normal order, and parses it to
the underlying FermionSequence() structure while maintaining the relation to the FermionOps.
"""

###############################################################################
# ## Compile in ordered basis

N = 9  # 9 spinful sites for 18 fermions

## Example: Built a Sympy hamiltonian and FermionOps
H_sympy, ops, params = build_spinful_chain_hamiltonian(N)
t, U, mu = params
H_sympy

""
## Initialize the FermionSequence object
H_seq = fock_space.FermionSequence(H_sympy, sympy_operators=ops)
H_seq

""
## Compile the data representation
H_data = H_seq.compile(data_format="coo")
data_coo = H_data.get()  ## Obtain the array of data
data_coo

###############################################################################
# ## Optional: compile in custom basis

## Make a custom basis
basis = fock_space.FockBasis(2 * N)  ## Create full basis for 18 fermionic sites
even_basis = basis.restrict(ParityMask("even"))  ## Restrict the basis using ParityMask
""
H_data = H_seq.compile(even_basis, data_format="coo")
H_data.get()

""
## Any FermionSequence() can be converted to sympy. Returns either the original operators, or creates them if origin was not a sympy expr
sympy_expr, operators = H_seq.to_sympy()
sympy_expr  ## This is now back to a sympy expression
###############################################################################
# ## Compact example

## Built sympy array
N = 7
H_sympy, ops, params = build_spinful_chain_hamiltonian(N)
t, U, mu = params
## Pass Sympy to FermionSequence
H = fock_space.FermionSequence(H_sympy, sympy_operators=ops)

## Compile in default fock basis, returning COO format
H_data = H.compile(data_format="coo", cached=True)

## Set parametrized values
H_data({t: 5.0, U: 2.5, mu: 3})

## Get data array
H_data.get()

###############################################################################
# # Examples/validation

###############################################################################
# ## Kitaev Chain

from example_models import kitaev_chain  # noqa: E402

""
## Setup Kitaev chain and basis
N = 4
H, ops, params = kitaev_chain(N)
t, Delta, mu = params

H_seq = fock_space.FermionSequence(H, sympy_operators=ops)
H_seq

###############################################################################
# ### Simple Block Diagonalization

## Sidenote: it is possible to request the block_diagonal basis after compiling once in any other basis:
N = 4
H, ops, params = kitaev_chain(N)
t, Delta, mu = params

H_seq = fock_space.FermionSequence(H, sympy_operators=ops)
H_data = H_seq.compile(data_format="array")
H_data({t: 12, Delta: 40, mu: 10})

block_diagonal_basis = H_data.to_block_diagonal_basis()
H_data_bd = H_seq.compile(block_diagonal_basis, data_format="array")
H_data_bd({t: 12, Delta: 40, mu: 10})

fig, axs = plt.subplots(ncols=2)
axs[0].set_title("Compile in ordered basis")
axs[1].set_title("Compile in block diagonal basis")
axs[0].matshow(np.array(H_data.get(), dtype=np.int64))
axs[1].matshow(np.array(H_data_bd.get(), dtype=np.int64))

###############################################################################
# ### Odd/even parity phase diagram
#
# Output parity phase diagram, using here the H_odd and H_even basis objects

full_basis = fock_space.FockBasis(N)
even_basis = full_basis.restrict(ParityMask("even"))
odd_basis = full_basis.restrict(ParityMask("odd"))

H_even = H_seq.compile(even_basis, data_format="array")
H_odd = H_seq.compile(odd_basis, data_format="array")
H_even({t: 0, Delta: 40, mu: 0})
H_odd({t: 0, Delta: 40, mu: 0})

""
mu_range = np.linspace(-100, 100, 101)
t_range = np.linspace(0, 100, 101)

result = np.empty((len(t_range), len(mu_range)))

for i, t_val in enumerate(t_range):
    # set t once
    H_odd({t: t_val})
    H_even({t: t_val})

    for j, mu_val in enumerate(mu_range):
        # set mu once
        H_odd({mu: mu_val})
        H_even({mu: mu_val})

        H_odd_mat = H_odd.get()
        H_even_mat = H_even.get()

        E_gs_odd = np.linalg.eigvalsh(H_odd_mat)[0]
        E_gs_even = np.linalg.eigvalsh(H_even_mat)[0]

        result[i, j] = np.sign(E_gs_odd - E_gs_even)

fig, axs = plt.subplots(1, figsize=(3, 3))
axs.pcolormesh(t_range, mu_range, result.T, cmap="RdBu_r")

###############################################################################
# ## Clock - Model

###############################################################################
# ### System is parametrized by non-linear terms

from example_models import clock_hamiltonian  # noqa: E402

""
N = 2
H_ferm, ops, phi = clock_hamiltonian(
    N=N,
)
H = fock_space.FermionSequence(H_ferm, sympy_operators=ops)
H

""
## Define restricted basis
color = ["black", "green", "blue"]
fig, axs = plt.subplots(1, figsize=(2.5, 2))
axs.set_xlabel("$\\phi$")
axs.set_ylabel("E")

states = fock_space.FockBasis(2 * N).restrict(DoubleOccupancyMask())
H_data = H.compile(states, data_format="array")

phase_range = np.linspace(0, 2 * np.pi, num=101)

# Pre-allocate storage per color
phi_vals = {c: [] for c in color}
E_vals = {c: [] for c in color}

for phi_val in phase_range:
    H_data.update_values({phi: phi_val})
    E, _ = np.linalg.eigh(H_data.get())

    for i in range(0, len(E), 3):
        c = color[(i // 3) % 3]
        phi_vals[c].extend([phi_val] * 3)
        E_vals[c].extend(E[i : i + 3])

# Now only 3 scatter calls instead of thousands
for c in color:
    axs.scatter(phi_vals[c], E_vals[c], c=c, s=10)

###############################################################################
# ## Fermi-Hubbard
