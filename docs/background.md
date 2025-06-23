# Theoretical Background: Fermionic Operators and the Jordan-Wigner Transformation

The `second_quantization` package is designed to facilitate symbolic and numerical calculations with fermionic operators in quantum many-body systems.
A key challenge in such systems is the non-local nature of fermionic creation and annihilation operators due to their anticommutation relations.
This non-locality is handled in our package using the **Jordan-Wigner transformation**.

## Fermionic Operators and Anticommutation

Fermionic creation ($c_i^\dagger$) and annihilation ($c_i$) operators satisfy the canonical anticommutation relations:

$$
{c_i, c_j} = 0, \qquad {c_i^\dagger, c_j^\dagger} = 0, \qquad {c_i, c_j^\dagger} = \delta_{ij}
$$

These relations mean that the order of operators is crucial, and that the operators are inherently non-local when mapped to a matrix representation.

## The Jordan-Wigner Transformation

To represent fermionic operators as matrices acting on a Hilbert space, we use the **Jordan-Wigner transformation**.
This transformation maps each fermionic operator to a string of Pauli matrices, introducing a string of $\sigma_z$ operators to ensure the correct anticommutation relations:

$$
c_j = I \otimes \cdots \otimes I \otimes \underbrace{\sigma_-}_{j\text{th site}} \otimes \sigma_z \otimes \cdots \otimes \sigma_z
$$

where $\sigma_- = (\sigma_x - i \sigma_y)/2$ is the lowering operator, and the string of $\sigma_z$ operators extends over all sites to the right of site $j$.

This mapping ensures that the resulting matrices obey the correct fermionic statistics, even though the underlying Hilbert space is a tensor product of local two-level systems.

## Implementation in `second_quantization`

In the `hilbert_space` module, the Jordan-Wigner transformation is used to construct the matrix representations of fermionic operators and their products. This allows the package to:

- Convert symbolic expressions involving fermionic operators into explicit matrices
- Perform algebraic manipulations and numerical calculations while preserving fermionic statistics
- Support both dense and sparse matrix representations for efficient computation

By leveraging the Jordan-Wigner transformation, the package provides a robust and general framework for working with second quantized Hamiltonians and observables in quantum many-body physics.
