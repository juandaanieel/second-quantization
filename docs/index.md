# Second Quantization

Second Quantization is an open-source Python package for fermionic, bosonic, and mixed many-body Hamiltonians.
It provides a symbolic interface built on SymPy's `FermionOp` and `BosonOp` for writing second-quantized operator expressions and converting them into dense or sparse numerical matrices. Bosonic modes require a finite occupation cutoff.

Second Quantization is part of a set of complementary numerical libraries for quantum many-body physics such as:

- [`qrate`](https://gitlab.kwant-project.org/qt/qrate) for transport calculations,

- [`pmm-tools`](https://gitlab.kwant-project.org/qt/pmm_tools) for simulation of one dimensional fermionic chains.

## Key features

- Simple symbolic construction: write Hamiltonians using fermionic and bosonic creation/annihilation operators and symbolic parameters.
- Bidirectional workflow: convert from symbolic operators to matrices (`to_matrix`) and recover symbolic operator decompositions from matrices (`to_operators`).
- Performance-oriented: supports sparse and dense formats, caching of basis operators, and integrates with numerical backends for efficient evaluation of large systems.

## Design principles

1. Clarity: expressive symbolic syntax closely matches pen-and-paper notation.
2. Reproducibility: parameterized, callable Hamiltonians enable reproducible sweeps and analyses.
3. Integration: work with existing numerical tools (OpenFermion, QuSpin, QuTiP) for fermionic, bosonic, and hybrid systems.

## Citation

If you use `second_quantization` in your research, and you want to acknowledge it please cite:

```bibtex
@misc{secondquantization,
  author       = {Second Quantization Developers},
  title        = {Second Quantization},
  howpublished = {\url{https://second-quantization.readthedocs.io/}},
  year         = {2025},
}
```

## Development

If you want to contribute to the development of `second_quantization`, or you find any problem, please [open an issue](https://gitlab.kwant-project.org/qt/second_quantization/-/issues?sort=created_date&state=opened&first_page_size=100).
