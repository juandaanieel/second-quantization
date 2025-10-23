---
execute: true
---

# Interacting Kitaev Chain

In this tutorial, we demonstrate how to use the `second_quantization` package to model and analyze an interacting Kitaev chain.
The Kitaev chain is a one-dimensional model of spinless fermions that exhibits topological superconductivity.
By introducing interactions nearest-neighbor Coulomb interactions, we can explore more complex phenomena such as many-body localization.

```python
# Symbolic tools
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger

# Numerical tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Second-quantization package
from second_quantization import hilbert_space
```

## Define Hamiltonian

The first step is to define the fermionic operators that will be used to construct the Hamiltonian.

```python
# Define the number of sites in the chain and fermionic operators
length = 8
names = ["c_{i}".format(i=i) for i in range(length)]
fermions = [FermionOp(name) for name in names]
```

Using `fermions`, we now define the Hamiltonian of the interacting Kitaev chain.

```python
# Define symbols for the Hamiltonian parameters
t, Delta, U = sympy.symbols("t Delta, U", real=True, positive=True)
mus = sympy.symbols(" ".join(f"mu_{i}" for i in range(length)), real=True)

# Define the interacting Kitaev chain Hamiltonian
H = t * sum(Dagger(a) * b for a, b in zip(fermions, fermions[1:]))  # Hopping
H += Delta * sum(Dagger(a) * Dagger(b) for a, b in zip(fermions, fermions[1:]))  # Pairing
H += Dagger(H)

for i, mu in enumerate(mus):
    H += mu * Dagger(fermions[i]) * fermions[i] # Chemical potential

H += U * sum(Dagger(a) * a * Dagger(b) * b for a, b in zip(fermions, fermions[1:])) # Nearest-neighbor Coulomb interaction
```

The total Hamiltonian is:

```python
display(sympy.Eq(sympy.Symbol("H"), H.factor()))
```

With both the fermionic operators and Hamiltonian defined, we can now proceed to build the matrix representation of the Hamiltonian.
The typical workflow of `second-quantization` involves two main steps:

- Converting the symbolic Hamiltonian into its matrix representation using `hilbert_space.to_matrix`.

- Creating a callable function that generates the Hamiltonian matrix for given parameter values using `hilbert_space.make_dict_callable`.

```python

terms = hilbert_space.to_matrix(H, operators=fermions, sparse=True)
f_hamiltonian = hilbert_space.make_dict_callable(terms)
```


## Diagonalize Hamiltonian with Parity Superselection Rules

Simulating quantum mechanical systems typically requires to find the eigenvalues and eigenvectors of the Hamiltonian.
In fermionic systems, the Hamiltonian exhibits a block-diagonal structure due to parity conservation.
Therefore, we can optimize the diagonalization process by separating the Hamiltonian into even and odd parity subspaces.

```python
parity_operator = hilbert_space.parity_operator(fermions, sparse=True)
```

```python
def fermionic_eigsh(hamiltonian, parity, k=6):
    """
    Diagonalize a Hamiltonian using fermionic superselection rules.
    The Hamiltonian is assumed to be block-diagonal in the parity
    subspaces. The function returns the eigenvalues and eigenvectors
    of the Hamiltonian, sorted by increasing energy.

    Parameters
    ----------
    hamiltonian : scipy.sparse.csr_matrix
        The Hamiltonian matrix to be diagonalized.
    parity : scipy.sparse.csr_matrix
        The parity operator matrix, which is assumed to be diagonal
        in the same basis as the Hamiltonian.
    k : int
        The number of eigenvalues and eigenvectors to compute.
        Default is 6.
    Returns
    -------
    evals : list of numpy.ndarray
        The eigenvalues of the Hamiltonian separated in parity subspaces.
    evecs : list of numpy.ndarray
        The eigenvectors of the Hamiltonian separated in parity subspaces.
    """
    # Separate the Hamiltonian into even and odd parity blocks
    par_mat = parity.diagonal()
    par_mat = 2*(par_mat-1/2)

    odd_inds, even_inds = (
        np.where(par_mat < 0)[0],
        np.where(par_mat > 0)[0],
    )

    h_odd = hamiltonian[odd_inds, :][:, odd_inds]
    h_even = hamiltonian[even_inds, :][:, even_inds]

    off_diagonal = hamiltonian[even_inds, :][:, odd_inds]
    assert off_diagonal.count_nonzero() == 0

    off_diagonal = hamiltonian[odd_inds, :][:, even_inds]
    assert off_diagonal.count_nonzero() == 0

    # Diagonalize each block separately
    evals_even, evecs_even = eigsh(h_even, k=k, which="SA", return_eigenvectors=True, v0=None)
    evals_odd, evecs_odd = eigsh(h_odd, k=k, which="SA", return_eigenvectors=True, v0=None)

    # Combine the eigenvectors into the full space
    new_evecs_even = np.zeros((par_mat.shape[0], evecs_even.shape[1]), dtype=complex)
    new_evecs_odd = np.zeros((par_mat.shape[0], evecs_odd.shape[1]), dtype=complex)

    new_evecs_even[even_inds, :] = evecs_even
    new_evecs_odd[odd_inds, :] = evecs_odd

    # Organise results
    evals = np.array([evals_even, evals_odd])
    evecs = np.array([new_evecs_even, new_evecs_odd])

    return evals, evecs
```

Now we can study the spectrum of the interacting Kitaev chain as a function of the interaction strength `U` and chemical potential `mu`.

```python
# Define parameters
t_val = 1.1
Delta_val = 2.2
mu_val = np.linspace(-3, 3, 25)

U_values = np.linspace(0, 4, 26)
energies = []

for U_val in U_values:
    for _mu in mu_val:
        param_dict = {'t': t_val, 'Delta': Delta_val, 'U': U_val}
        for i, mu_sym in enumerate(mus):
            param_dict[str(mu_sym)] = _mu

        H_matrix = f_hamiltonian(**param_dict)
        evals, _ = fermionic_eigsh(H_matrix, parity_operator, k=8)
        energies.append(evals)

energies = np.array(energies).reshape(len(U_values), len(mu_val), 2, -1)
gs_energy = energies[:, :, 1, 0] - energies[:, :, 0, 0]
```
Finally, we can visualize the low-energy spectrum of the interacting Kitaev chain.

```python
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(
    U_values,
    mu_val,
    gs_energy.T,
    shading="auto",
    cmap="viridis",
)

ax.set_xlabel("Interaction Strength U")
ax.set_ylabel("Chemical Potential μ")
ax.set_title("Energy Gap between Ground and First Excited State")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Energy Gap")
plt.show()
```
