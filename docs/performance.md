---
execute: true
---

# Performance

In this tutorial, we will compare the performance of `second-quantization` with two other popular many-body quantum simulation libraries:

- [`OpenFermion`](https://github.com/quantumlib/OpenFermion)
- [`QuSpin`](https://quspin.github.io/QuSpin/)

In particular, we will compare the time taken to build the Hamiltonian of a chain of fermions.
Because our package focuses on finding the matrix representation of a symbolic Hamiltonian, the time required to diagonalise the Hamiltonian is not considered here.

## Setup

1. First we import the necessary libraries.

```python
import time
from timeit import timeit
import numpy as np
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import matplotlib.pyplot as plt

from second_quantization import hilbert_space

from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import hamiltonian

import openfermion
from openfermion.transforms import get_sparse_operator
from openfermion.ops import FermionOperator
```

2. We define a function to build the Hamiltonian for a chain of spinful fermions using `second-quantization`.

```python
def build_spinful_chain_hamiltonian(n):
    """
    Builds a chain Hamiltonian for spinful fermions with:
    - nearest-neighbor hopping for both spins,
    - onsite Coulomb interaction between up/down spins,
    - chemical potential for both spins.
    There are 2*n fermionic modes: c_{i,up}, c_{i,down} for i in 0..n-1.
    """
    t, U, mu = sympy.symbols('t U mu', real=True)
    ops_up = [FermionOp(fr"c_{{{i},\uparrow}}") for i in range(n)]
    ops_dn = [FermionOp(fr"c_{{{i},\downarrow}}") for i in range(n)]
    hopping = 0
    interaction = 0
    chemical = 0

    for i in range(n - 1):
        # Hopping for up
        hopping += -t * (Dagger(ops_up[i]) * ops_up[i + 1] + Dagger(ops_up[i + 1]) * ops_up[i])
        # Hopping for down
        hopping += -t * (Dagger(ops_dn[i]) * ops_dn[i + 1] + Dagger(ops_dn[i + 1]) * ops_dn[i])

    for i in range(n):
        # Onsite Coulomb interaction
        interaction += U * (Dagger(ops_up[i]) * ops_up[i]) * (Dagger(ops_dn[i]) * ops_dn[i])
        # Chemical potential for both spins
        chemical += -mu * (Dagger(ops_up[i]) * ops_up[i] + Dagger(ops_dn[i]) * ops_dn[i])

    H = hopping + interaction + chemical
    ops = ops_up + ops_dn
    return H, ops
```

We illustrate the Hamiltonian for a small system with `n=2` sites.

```python
H, ops = build_spinful_chain_hamiltonian(2)
sympy.Eq(sympy.Symbol('H'), H.factor())
```

3. We define the corresponding functions for `QuSpin` and `OpenFermion`.

```python
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
        H += -t * (FermionOperator(f'{2*i}^ {2*(i+1)}') + FermionOperator(f'{2*(i+1)}^ {2*i}'))
        # Hopping for down
        H += -t * (FermionOperator(f'{2*i+1}^ {2*(i+1)+1}') + FermionOperator(f'{2*(i+1)+1}^ {2*i+1}'))
    for i in range(n):
        # Onsite Coulomb interaction
        H += U * FermionOperator(f'{2*i}^ {2*i} {2*i+1}^ {2*i+1}')
        # Chemical potential for both spins
        H += -mu * (FermionOperator(f'{2*i}^ {2*i}') + FermionOperator(f'{2*i+1}^ {2*i+1}'))
    return H


def quspin_sparse_hamiltonian(L):
    """
    Builds a sparse Hamiltonian for a 1D spinful fermion model.
    """
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
```

## Benchmarking

We will benchmark two aspects of the Hamiltonian construction:

- Time required to setup the simulation. This includes building the basis and Hamiltonian representation, but not constructing the sparse matrix.

- Time required to build the sparse matrix representation of the Hamiltonian.

```python tags=["hide-cell"]
sparse_times_quspin = []
sparse_times_sq = []
sparse_times_of = []

basis_times_sq = []
basis_times_quspin = []
basis_times_of = []
```

```python tags=["hide-output"]
max_n = 10
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
    start_of = time.time()
    H = build_spinful_chain_hamiltonian_openfermion(n)
    basis_times_of.append(time.time() - start_of)

    sparse_time_of = time.time()
    H_sparse_openfermion = get_sparse_operator(H)
    sparse_times_of.append(time.time() - sparse_time_of)
```

## Results

We plot the time taken to setup up the simulation and to build the sparse matrix representation of the Hamiltonian as a function of the number of fermions.

```python tags=["hide-input"]
fig, axs = plt.subplots(ncols=2, figsize=(7, 3))

ns = list(range(2, max_n + 1))
ns = 2 * np.array(ns)
# Panel 1: Matrix build times
axs[0].plot(ns, sparse_times_sq, 'o--', label='SQ (to_matrix)')
axs[0].plot(ns, sparse_times_quspin, 'o-.', label='QuSpin')
axs[0].plot(ns, sparse_times_of, 'o-', label='OpenFermion')
axs[0].set_yscale('log')
axs[0].set_xlabel('Number of fermions')
axs[0].set_ylabel('Time (s)')
axs[0].set_title('Build sparse matrix')
axs[0].legend(frameon=False)

# Panel 2: Basis build times
axs[1].plot(ns, basis_times_sq, 'o--', label='SQ (basis_operators)')
axs[1].plot(ns, basis_times_quspin, 'o-.', label='QuSpin')
axs[1].plot(ns, basis_times_of, 'o-', label='OpenFermion')
axs[1].set_yscale('log')
axs[1].set_xlabel('Number of fermions')
axs[1].set_ylabel('Time (s)')
axs[1].set_title('Setup required structures')

plt.tight_layout()
plt.show()
```

## Reusability

An important aspect of `second-quantization` is that the basis operators can be reused to build multiple Hamiltonians without recomputing the basis each time.
After creating the Hamiltonian once, creating an instance with a different parameter configuration takes very little time:

```python
H, ops = build_spinful_chain_hamiltonian(n)
H_dict = hilbert_space.to_matrix(H, operators=ops, sparse=True)
f_H = hilbert_space.make_dict_callable(H_dict)

# total time for 100 runs
t = timeit(stmt='f_H(U=1, t=1, mu=.1)', number=100, globals=globals())
print("Time per-call (s):", t / 100)
```
