# second_quantization

```{toctree}
:maxdepth: 2
:caption: Contents:

tutorial/index
api
```

## Converting second quantized operators to matrix form and back

The `second_quantization` package provides tools for working with second quantized operators in quantum many-body systems using `sympy.physics.quantum`.
It allows you to:
- Combine symbolic operations on second quantized operators with numerical matrix representations.
- Convert back and forth between symbolic expressions and numerical matrices.
- Perform algebraic manipulations of fermionic operators such as the fermionic partial trace.

## Quick Start

Install the package and start exploring quantum many-body systems.
The main module in `second_quantization` is `hilbert_space`.
Here we illustrate how to define a simple Hamiltonian and convert it to matrix form.

```python
import sympy
from sympy.physics.quantum import FermionOp
from sympy.physics.quantum import Dagger

from second_quantization import hilbert_space

# Define symbols and fermionic operators
mu, t, U = sympy.symbols("mu, t, U", real=True, commutative=True)
fermions = FermionOp("c_0"), FermionOp("c_1")
c_0, c_1 = fermions

# Define the Hamiltonian for a simple two-site fermionic system
hamiltonian = (
    -t * (Dagger(c_0) * c_1 + Dagger(c_1) * c_0)
    + mu * (Dagger(c_0) * c_0 + Dagger(c_1) * c_1)
    + U * Dagger(c_0) * c_0 * Dagger(c_1) * c_1
)

# Obtain the matrix representation of the Hamiltonian
matrix_form = hilbert_space.to_matrix(hamiltonian, operators=fermions, sparse=False)
f_hamiltonian = hilbert_space.make_dict_callable(hamiltonian)
```

## Tutorials

Check out our comprehensive tutorials:

- **Tutorial Overview**: [tutorial/index](tutorial/index.md) - Introduction to the package capabilities

## API Reference

For detailed API documentation, see the [api](api.md) reference.
