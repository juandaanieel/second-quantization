---
execute: true
---

# Interacting Kitaev Chain

[The Kitaev chain](https://iopscience.iop.org/article/10.1070/1063-7869/44/10S/S29) is a landmark model in condensed matter physics, describing a one-dimensional system of spinless fermions that exhibits **topological superconductivity**. In its non-interacting form, the model supports Majorana zero modes at its boundaries—exotic quasiparticles that are promising candidates for topological quantum computing.

In this tutorial, we extend the standard Kitaev chain by adding **nearest-neighbor Coulomb interactions**. This modification breaks the model's exact solvability and enables the exploration of rich many-body physics, including interaction-driven phase transitions and correlation effects.

We'll demonstrate how to:

1. Construct the symbolic Hamiltonian using fermionic operators
2. Convert it to matrix form with `second_quantization`
3. Leverage parity conservation for efficient diagonalization
4. Compute phase diagrams as a function of different system parameters

Let's get started!

```python
# Symbolic tools
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from IPython.display import display

# Numerical tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Second-quantization package
from second_quantization import hilbert_space
```

## Constructing the Hamiltonian

### Step 1: Define Fermionic Operators

We begin by creating fermionic annihilation operators $c_i$ for each site in our chain:

```python
# Define the number of sites in the chain and fermionic operators
length = 6
names = ["c_{i}".format(i=i) for i in range(length)]
fermions = [FermionOp(name) for name in names]
```

### Step 2: Build the Symbolic Hamiltonian

The interacting Kitaev Hamiltonian includes four key terms:

1. **Hopping** ($t$): Nearest-neighbor kinetic energy
2. **Superconducting pairing** ($\Delta$): Anomalous p-wave pairing that induces superconductivity
3. **Chemical potential** ($\mu_i$): Controls the energy at each site
4. **Coulomb interaction** ($U$): Nearest-neighbor density-density repulsion

Let's construct each term symbolically:

```python
# Define symbols for the Hamiltonian parameters
t, Delta, U = sympy.symbols("t Delta, U", real=True, positive=True)
mus = sympy.symbols(" ".join(f"mu_{i}" for i in range(length)), real=True)

# Hopping term: -t Σ (c†ᵢ cᵢ₊₁ + h.c.)
H = t * sum(Dagger(a) * b for a, b in zip(fermions, fermions[1:]))

# Pairing term: Δ Σ (c†ᵢ c†ᵢ₊₁ + h.c.)
H += Delta * sum(Dagger(a) * Dagger(b) for a, b in zip(fermions, fermions[1:]))

# Add Hermitian conjugate for hopping and pairing
H += Dagger(H)

# Chemical potential: Σ μᵢ c†ᵢ cᵢ

for i, mu in enumerate(mus):
    H += mu * Dagger(fermions[i]) * fermions[i]

# Interaction term: U Σ nᵢ nᵢ₊₁ where nᵢ = c†ᵢ cᵢ
H += U * sum(Dagger(a) * a * Dagger(b) * b for a, b in zip(fermions, fermions[1:]))
```

Let's examine the complete Hamiltonian in factored form:

```python
display(sympy.Eq(sympy.Symbol("H"), H.factor()))
```

### Step 3: Convert to Matrix Form

Now comes the key step: transforming our symbolic Hamiltonian into a numerical matrix. The `second_quantization` workflow involves:

1. **`to_matrix`**: Maps fermionic operators to their matrix representations using the Jordan-Wigner transformation
2. **`make_dict_callable`**: Creates a callable function that efficiently generates Hamiltonian matrices for any parameter values

```python
# Build the symbolic-to-matrix mapping
terms = hilbert_space.to_matrix(H, operators=fermions, sparse=True)

# Create a callable function H(t, Delta, U, mu_0, ..., mu_7)
f_hamiltonian = hilbert_space.make_dict_callable(terms)
```

This approach separates the (expensive) symbolic processing from the (cheap) numerical evaluation, making parameter sweeps extremely efficient.

## Efficient Diagonalization with Parity Conservation

### Exploiting Symmetry

Fermionic systems conserve **fermion parity**—the total number of fermions modulo 2. This means the Hamiltonian is block-diagonal in the parity basis:

```python
parity_operator = hilbert_space.parity_operator(fermions, sparse=True)
```

By diagonalizing the even and odd parity blocks separately, we can:

- **Reduce computational cost** by working with smaller matrices
- **Compute ground states** in specific parity sectors

Here's a function that performs parity-resolved diagonalization:

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

## Computing the Phase Diagram

Now we'll compute how the energy gap between the ground state and first excited state varies with interaction strength $U$ and chemical potential $\mu$. This gap is crucial for understanding topological protection and phase transitions.

```python
# Define parameters
t_val = 1.1
mu_val = np.linspace(0, 6, 55)

U_values = np.linspace(-4, 4, 56)
energies = []

for U_val in U_values:
    Delta_val = t_val + U_val / 2  # Example relation between Delta and U
    for _mu in mu_val:
        # Build parameter dictionary
        param_dict = {'t': t_val, 'Delta': Delta_val, 'U': U_val}

        # Set site-dependent chemical potentials (accounting for interaction shift)
        for i, mu in enumerate(mus):
            if i > 0 and i < length - 1:
                # Bulk sites: full interaction correction
                param_dict[str(mu)] = _mu - U_val
            else:
                # Edge sites: half interaction correction
                param_dict[str(mu)] = _mu - U_val / 2

        # Generate Hamiltonian and diagonalize
        H_matrix = f_hamiltonian(**param_dict)
        evals, _ = fermionic_eigsh(H_matrix, parity_operator, k=8)
        energies.append(evals)

# Reshape and compute gap
energies = np.array(energies).reshape(len(U_values), len(mu_val), 2, -1)
gs_energy = energies[:, :, 1, 0] - energies[:, :, 0, 0]
```

### Visualization

Finally, we can visualize the low-energy spectrum of the interacting Kitaev chain.

```python
fig, ax = plt.subplots(figsize=(4, 4))
im = ax.pcolormesh(
    U_values,
    mu_val,
    np.abs(gs_energy.T),
    shading="auto",
    cmap="viridis",
    vmin=0,
    vmax=2
)

ax.set_xlabel(r"$U$")
ax.set_ylabel(r"$\mu$")
ax.set_title(r"$E_{\text{gap}}$")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Energy Gap")
plt.show()
```

## Summary
This tutorial demonstrates how `second_quantization` makes it straightforward to:

1. Build complex many-body Hamiltonians symbolically
2. Leverage symmetries for efficient computation
3. Explore phase diagrams across parameter space

The same workflow applies to arbitrary fermionic systems—from Hubbard models to quantum dots to topological superconductors!
