# Second quantization

## Direct plan

* (sympy) expression in terms of creation and annihilation operators/states <-> numerical/symbolical matrix/vectors
* fermions, (truncated) bosons, spins and mixtures of those
* Consider typical performance and tradeoffs; sparse + 100k hilbert space reasonable
* Making this a small package

## Later/related

* Represent lattice operators $\sum_\mathbf{n,k} c^\dagger_{n+a,k + \delta k}c_{n,k}$
* Consider moving to sympy and contributing to it

## Procedure

### CAOs to matrices

1. We define a finite basis of second quantized operators using `FermionOp` from `sympy`.
2. We express the operators as a Jordan-Wigner string, i.e.

$$
c_{i} = \left[ \Pi_{j=0}^{i-1} \sigma_z^{j}  \right] \times \sigma_{-}^{i} \times \left[ \Pi_{j=0}^{i-1} \sigma_0^{j}  \right].
$$

3. We use the matrix representation of each CAO in order to build products, e.g. $c_{1}^\dagger c_1$.


### Operators in matrix form to CAOs

1. We transform the matrix to a Pauli string using [this code](https://gitlab.kwant-project.org/qt/time-dependent-pt/-/blob/tunable-coupler-note/codes/pauli_decomposition.py).
2. We identify the different terms in the Pauli string using the first three equations [here](https://en.wikipedia.org/wiki/Jordan–Wigner_transformation).

### Coding aspect

We want to develop the following functions:

```python

def basis_operators(
    operators: list[sympy.Expr],
    fermion_ordering: list[sympy.physics.quantum.Fermion]
    )
    -> dict[sympy.expr, numpy.ndarray | scipy.sparse.array]:

def to_matrix(
    expr: sympy.expr
    basis_operators: [sympy.expr, numpy.ndarray | scipy.sparse.array]
    )
    -> dict[sympy.expr, numpy.ndarray | scipy.sparse.array]:

def to_operators(
    matrix: numpy.ndarray | scipy.sparse.array,
    operators: list[sympy.Expr],
    fermion_ordering: list[sympy.physics.quantum.Fermion]
    )
    -> sympy.expr:

```

### Tests

We want to test the code.
Some possible things to test are:

- Checking input and output types
- Idempotence (going back and forth should do nothing)
- Preserving algebraic properties