# 2D Hubbard Model Example

This example demonstrates how to use the `second_quantization` package to construct and analyze a 2D Hubbard model with spinful fermions and local Coulomb interaction.

## Model Description

The Hubbard model on a 2D lattice is a paradigmatic model for strongly correlated electrons. The Hamiltonian is:

$$
H = -t \sum_{\langle i,j \rangle,\sigma} (c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.}) + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

where:
- $t$ is the hopping amplitude between nearest neighbors
- $U$ is the on-site Coulomb repulsion
- $c_{i\sigma}^\dagger$ ($c_{i\sigma}$) creates (annihilates) a fermion with spin $\sigma$ on site $i$
- $n_{i\sigma} = c_{i\sigma}^\dagger c_{i\sigma}$ is the number operator

## Setup

```python
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp
from second_quantization import hilbert_space

# Lattice size (2x2)
Lx, Ly = 2, 2
N_sites = Lx * Ly
spins = ['up', 'down']

# Symbolic parameters
U, t = sympy.symbols('U t', real=True)

# Helper to get site index
site_index = lambda x, y: x + Lx * y

# Create fermionic operators for each site and spin
fermions = {}
for x in range(Lx):
    for y in range(Ly):
        for s in spins:
            label = f"c_{x}_{y}_{s}"
            fermions[(x, y, s)] = FermionOp(label)

# List of all operators (order matters for basis)
operators = [fermions[(x, y, s)] for x in range(Lx) for y in range(Ly) for s in spins]

# Hopping terms (nearest neighbors)
hopping = 0
for x in range(Lx):
    for y in range(Ly):
        for dx, dy in [(1, 0), (0, 1)]:  # right and up neighbors
            nx, ny = (x + dx) % Lx, (y + dy) % Ly  # periodic BCs
            for s in spins:
                c1 = fermions[(x, y, s)]
                c2 = fermions[(nx, ny, s)]
                hopping += -t * (Dagger(c1) * c2 + Dagger(c2) * c1)

# On-site interaction terms
interaction = 0
for x in range(Lx):
    for y in range(Ly):
        n_up = Dagger(fermions[(x, y, 'up')]) * fermions[(x, y, 'up')]
        n_down = Dagger(fermions[(x, y, 'down')]) * fermions[(x, y, 'down')]
        interaction += U * n_up * n_down

# Total Hamiltonian
H = hopping + interaction
print("Symbolic Hamiltonian:")
print(H)

# Convert to matrix representation
H_matrix = hilbert_space.to_matrix(H, operators=operators, sparse=False)
print("\nHamiltonian matrix shape:", H_matrix[1].shape)
```

## Analysis

- The Hilbert space dimension is $2^{N_{\text{sites}} \times 2}$ (spin up/down per site).
- For $2\times2$ sites, the dimension is $2^8 = 256$.
- You can diagonalize `H_matrix[1]` with `numpy.linalg.eigh` to obtain the spectrum.

---

This example can be extended to larger lattices, open boundary conditions, or additional terms (e.g., chemical potential, magnetic field) as needed.
