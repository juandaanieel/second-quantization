# API documentation

## Migration from the fermion-only API

- Import `PauliDecomposition` from `second_quantization.fermions` instead of
  `second_quantization.pauli_strings`.
- `hilbert_space.basis_operators` returns `(fermion_dict, boson_dict)`. Pass this
  pair unchanged as `operator_dict` when reusing it in `to_matrix`.
- Matrix conversion, `symbolic_basis`, and partial traces place fermions before
  bosons, preserving the input order within each type. The rightmost mode varies
  fastest.
- Bosonic truncation `t` includes occupations `0` through `t`, giving `t + 1`
  states. An integer applies to every boson; a list follows the bosons' input
  order. Fermion-only calls need no truncation.
- Parity is `0` for even and `1` for odd total occupation for fermions, bosons,
  and mixtures. This preserves the fermion-only convention.

Symbolic parameters can be kept as dictionary keys for all particle types.
Pauli decomposition also accepts matrices with symbolic entries. Bosonic
inversion operates on numerical matrices and drops coefficients at or below
`1e-10` by default; it reconstructs operators within the chosen finite cutoff.

## Hilbert Space
::: second_quantization.hilbert_space

## Fermions
::: second_quantization.fermions

## Bosons
::: second_quantization.bosons
