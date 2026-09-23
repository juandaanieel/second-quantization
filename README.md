# Second Quantization

[![PyPI version](https://badge.fury.io/py/second-quantization.svg)](https://badge.fury.io/py/second-quantization)
[![Tests](https://gitlab.kwant-project.org/qt/second_quantization/badges/main/pipeline.svg)](https://gitlab.kwant-project.org/qt/second_quantization/-/pipelines)

A Python package for seamless conversion between symbolic second-quantized operators and their numerical matrix representations.

## Features

- **Symbolic to Matrix**: Convert fermionic, truncated bosonic, and mixed operator expressions to matrices
- **Matrix to Symbolic**: Decompose matrices back into normal-ordered operator expressions
- **Efficient**: Supports both dense and sparse matrix representations for large Hilbert spaces
- **Flexible**: Handle parameterized Hamiltonians with symbolic coefficients
- **Fast Parameter Sweeps**: Reuse basis operators for different parameter values

## Installation

```bash
pip install second-quantization
```

## Quick Start

```python
import sympy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from second_quantization import hilbert_space

# Define fermionic operators
c, d = [FermionOp(name) for name in "cd"]

# Create a simple Hamiltonian: t(c†d + d†c) + U c†c d†d
t, U = sympy.symbols("t U", real=True)
H = t * (Dagger(c) * d + Dagger(d) * c) + U * Dagger(c) * c * Dagger(d) * d

# Convert to matrix representation
matrix_dict = hilbert_space.to_matrix(H, operators=[c, d], sparse=False)

# Create a callable function for parameter sweeps
hamiltonian_func = hilbert_space.make_dict_callable(matrix_dict)

# Generate Hamiltonian for specific parameter values
H_matrix = hamiltonian_func(t=1.0, U=2.5)
```

## Documentation

Full documentation with tutorials and API reference is available at:
[https://qt.pages.quantumtinkerer.group/second_quantization/](https://qt.pages.quantumtinkerer.group/second_quantization/)

## Use Cases

- **Quantum Many-Body Physics**: Study interacting fermion systems
- **Quantum Chemistry**: Molecular Hamiltonians and electronic structure
- **Condensed Matter**: Hubbard models, topological systems, quantum dots
- **Quantum Computing**: Variational quantum algorithms, QAOA
- **Education**: Learn fermionic second quantization interactively

## Core Functions

### `hilbert_space.to_matrix()`
Convert fermionic and bosonic expressions to matrices using Jordan-Wigner and truncated occupation bases.

### `hilbert_space.make_dict_callable()`
Create efficient callable functions from symbolic Hamiltonians for parameter sweeps.

### `hilbert_space.to_operators()`
Decompose matrices back into symbolic fermionic, bosonic, or mixed operator expressions.

### `hilbert_space.basis_operators()`
Generate fermionic and bosonic matrix bases, returned as a pair of dictionaries.

## Requirements

- Python ≥ 3.11
- NumPy ≥ 2.2
- SciPy ≥ 1.14
- SymPy ≥ 1.13

## Contributing

Contributions are welcome! Please see our [repository](https://gitlab.kwant-project.org/qt/second_quantization) for development setup and contribution guidelines.

## License

BSD 2-Clause License. See [LICENSE](LICENSE) for details, including the MIT notice for the adapted Pauli decomposition.
