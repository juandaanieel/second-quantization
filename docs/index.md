# Second Quantization

Second Quantization is an open-source Python package that simplifies working with fermionic many-body Hamiltonians.
It provides an intuitive symbolic interface (built on SymPy's `FermionOp`) for writing second-quantized operator expressions and robust utilities to convert these expressions into numerical matrix representations suitable for dense and sparse linear-algebra workflows.

Second Quantization is part of a set of complementary numerical libraries for quantum many-body physics such as:

- `qrate` for transport calculations,

- `pmm-tools` for simulation of one dimensional fermionic chains.

## Key features

- Simple symbolic construction: write Hamiltonians using familiar fermionic creation/annihilation operators and symbolic parameters.
- Bidirectional workflow: convert from symbolic operators to matrices (`to_matrix`) and recover symbolic operator decompositions from matrices (`to_operators`).
- Performance-oriented: supports sparse and dense formats, caching of basis operators, and integrates with numerical backends for efficient evaluation of large systems.

## Design principles

1. Clarity: expressive symbolic syntax closely matches pen-and-paper notation.
2. Reproducibility: parameterized, callable Hamiltonians enable reproducible sweeps and analyses.
3. Extensibility: built to integrate with existing numerical tools (OpenFermion, QuSpin) and to be extended with bosonic degrees of freedom for hybrid systems.
