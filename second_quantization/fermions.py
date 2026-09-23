import numpy as np
import sympy
import scipy
from functools import reduce
from sympy.core.singleton import S
from sympy.physics.quantum import Dagger

sigma_0 = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_minus = np.array([[0, 1], [0, 0]])


# Matrix construction


def fermion_basis(
    operators: list[sympy.Expr], sparse: bool
) -> dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array :]:
    """Build Jordan–Wigner annihilation matrices.

    Parameters
    ----------
    operators : list[sympy.Expr]
        Fermionic modes in tensor-product order.
    sparse : bool
        Return CSR arrays instead of dense arrays.

    Returns
    -------
    dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]
        Full-space identity and one annihilation matrix per mode.
    """

    if sparse:

        def kron(a, b):
            return scipy.sparse.kron(a, b, format="csr")

        def eye(N):
            return scipy.sparse.eye(N, format="csr")
    else:
        kron = np.kron
        eye = np.eye

    size = len(operators)
    basis_operators = {1: eye(2**size)}
    for i, operator in enumerate(operators):
        basis_operators[operator] = reduce(
            kron, [sigma_z] * i + [sigma_minus] + [eye(2 ** (size - i - 1))]
        )

    return basis_operators


def fermion_parity(
    operators: list[sympy.Expr], sparse: bool
) -> np.ndarray | scipy.sparse.csr_array:
    """Build the fermionic occupation-parity matrix.

    Parameters
    ----------
    operators : list[sympy.Expr]
        Fermionic modes defining the register.
    sparse : bool
        Return a CSR array instead of a dense array.

    Returns
    -------
    np.ndarray | scipy.sparse.csr_array
        Diagonal matrix with 0 for even and 1 for odd occupation.
    """
    size = 2 ** len(operators)
    parity_diag = np.bitwise_count(np.arange(size)) % 2
    if sparse:
        return scipy.sparse.diags(parity_diag, format="csr", dtype=parity_diag.dtype)
    else:
        return np.diag(parity_diag)


# Matrix inversion


def _is_negligible(matrix, threshold):
    if isinstance(matrix, sympy.MatrixBase):
        return all((abs(value) <= threshold) is S.true for value in matrix)
    if scipy.sparse.issparse(matrix):
        matrix = matrix.data
    return np.all(np.abs(matrix) <= threshold)


# The recursive block decomposition in _pauli_coefficients and PauliDecomposition
# is adapted from Lukas Hantzko's https://github.com/HANTLUK/PauliDecomposition.
# Copyright (c) 2023 Lukas Hantzko. MIT license; full notice in LICENSE.
def _pauli_coefficients(matrix, depth, threshold=0.0):
    """Decompose leading two-level tensor factors into Pauli blocks.

    Parameters
    ----------
    matrix : np.ndarray | scipy.sparse.spmatrix | sympy.Matrix
        Matrix containing ``depth`` leading two-level factors.
    depth : int
        Number of factors to decompose.
    threshold : float
        Omit blocks whose maximum absolute entry is not larger than this value.
        Retain symbolic entries whose magnitude cannot be determined.

    Returns
    -------
    dict[str, np.ndarray | scipy.sparse.spmatrix | sympy.Matrix]
        Pauli strings mapped to their residual coefficient blocks.
    """
    imaginary = sympy.I if isinstance(matrix, sympy.MatrixBase) else 1j
    result = {}

    def decompose(block, prefix, remaining):
        if remaining == 0:
            if prefix or not _is_negligible(block, threshold):
                result[prefix] = block
            return

        half = block.shape[0] // 2
        top_left = block[:half, :half]
        top_right = block[:half, half:]
        bottom_left = block[half:, :half]
        bottom_right = block[half:, half:]
        children = (
            ("I", (top_left + bottom_right) / 2),
            ("X", (bottom_left + top_right) / 2),
            ("Y", -imaginary * (bottom_left - top_right) / 2),
            ("Z", (top_left - bottom_right) / 2),
        )
        for label, child in children:
            if not _is_negligible(child, threshold):
                decompose(child, prefix + label, remaining - 1)

    decompose(matrix, "", depth)
    return result


def PauliDecomposition(
    matrix: np.ndarray | scipy.sparse.spmatrix | sympy.Matrix,
    threshold: float = 0.0,
) -> dict[str, complex | float | sympy.Basic]:
    """Decompose a matrix into Pauli-string coefficients.

    Parameters
    ----------
    matrix : np.ndarray | scipy.sparse.spmatrix | sympy.Matrix
        Square matrix with power-of-two dimension.
    threshold : float
        Omit coefficients whose absolute value is not larger than this value.
        Retain symbolic coefficients whose magnitude cannot be determined.

    Returns
    -------
    dict[str, complex | float | sympy.Basic]
        Pauli strings mapped to scalar coefficients.

    Raises
    ------
    ValueError
        If ``matrix`` is not square or its dimension is not a power of two.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix is not square.")
    size = matrix.shape[0]
    if size == 0 or size & (size - 1):
        raise ValueError("Matrix dimension is not a power of 2.")
    blocks = _pauli_coefficients(matrix, size.bit_length() - 1, threshold)
    return {key: block[0, 0] for key, block in blocks.items()}


def string_to_fermion_operators(pauli_str: str, fermion_ops: list) -> sympy.Expr:
    """Convert a Pauli string through the inverse Jordan–Wigner map.

    Parameters
    ----------
    pauli_str : str
        String containing ``I``, ``X``, ``Y``, and ``Z`` labels.
    fermion_ops : list[sympy.Expr]
        Fermionic modes corresponding to the string positions.

    Returns
    -------
    sympy.Expr
        Fermionic operator expression before normal ordering.
    """
    expr = S.One
    parity = S.One

    for key, op in zip(pauli_str, fermion_ops):
        if key == "X":
            expr = expr * parity * (Dagger(op) + op)
        elif key == "Y":
            expr = expr * parity * (1j * (Dagger(op) - op))
        elif key == "Z":
            expr = expr * (-(2 * Dagger(op) * op - 1))
        parity = parity * (-(2 * Dagger(op) * op - 1))

    return expr
