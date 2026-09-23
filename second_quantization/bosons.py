import numpy as np
import scipy
import sympy
from functools import cache, partial, reduce
from itertools import pairwise
from math import factorial, prod, sqrt
from sympy.physics.quantum import Dagger


# Matrix construction


def _truncation_list(truncation, n_modes):
    if n_modes == 0:
        return []
    if truncation is None:
        raise ValueError("Please provide a truncation for your bosons")
    values = [truncation] * n_modes if isinstance(truncation, int) else list(truncation)
    if len(values) != n_modes:
        raise ValueError("Incompatible truncation and operator list lengths")
    return values


def boson_basis(
    operators: list[sympy.Expr], truncation: list[int] | int, sparse: bool
) -> dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]:
    """Build truncated bosonic annihilation-power matrices.

    Parameters
    ----------
    operators : list[sympy.Expr]
        Bosonic modes in tensor-product order.
    truncation : list[int] | int
        Per-mode cutoffs; a scalar applies to every mode.
    sparse : bool
        Return CSR arrays instead of dense arrays.

    Returns
    -------
    dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]
        Full-space matrices for powers zero through each mode's cutoff.

    Raises
    ------
    ValueError
        If the number of cutoffs does not match the number of modes.
    """
    truncation = _truncation_list(truncation, len(operators))

    if sparse:
        kron = partial(scipy.sparse.kron, format="csr")
        eye = partial(scipy.sparse.eye, format="csr")
    else:
        kron = np.kron
        eye = np.eye

    def annihilation_powers(dim: int, max_power: int):
        identity = eye(dim)
        powers = [identity]
        if max_power == 0:
            return powers

        factors = np.ones(dim, dtype=float)
        for p in range(1, max_power + 1):
            factors[p:] *= np.sqrt(np.arange(1, dim - p + 1, dtype=float))
            data = factors[p:].copy()
            if sparse:
                mat = scipy.sparse.diags([data], [p], shape=(dim, dim), format="csr")
            else:
                mat = np.diag(data, k=p)
            powers.append(mat)
        return powers

    boson_basis = {}

    identity_matrices = [eye(truncation[n] + 1) for n in range(len(operators))]

    for i, operator in enumerate(operators):
        base_operators = identity_matrices.copy()

        powers = annihilation_powers(truncation[i] + 1, truncation[i])
        for power in range(truncation[i] + 1):
            base_operators[i] = powers[power]
            boson_basis[operator**power] = reduce(kron, base_operators)
            base_operators[i] = identity_matrices[i]

    return boson_basis


def boson_parity(
    operators: list[sympy.Expr], truncation: list[int] | int, sparse: bool
) -> np.ndarray | scipy.sparse.csr_array:
    """Build the bosonic occupation-parity matrix.

    Parameters
    ----------
    operators : list[sympy.Expr]
        Bosonic modes in tensor-product order.
    truncation : list[int] | int
        Per-mode cutoffs; a scalar applies to every mode.
    sparse : bool
        Return a CSR array instead of a dense array.

    Returns
    -------
    np.ndarray | scipy.sparse.csr_array
        Diagonal matrix with 0 for even and 1 for odd total occupation.

    Raises
    ------
    ValueError
        If the number of cutoffs does not match the number of modes.
    """
    truncation = _truncation_list(truncation, len(operators))

    dims = np.array([t + 1 for t in truncation])
    total_dim = np.prod(dims)

    indices = np.arange(total_dim)

    total_occupation = np.zeros(total_dim, dtype=int)
    temp_indices = indices.copy()

    # The rightmost Kronecker factor is the fastest-varying index.
    for dim in dims[::-1]:
        occupation = temp_indices % dim
        total_occupation += occupation
        temp_indices //= dim
    parity_diag = total_occupation % 2

    if sparse:
        return scipy.sparse.diags(parity_diag, format="csr", dtype=parity_diag.dtype)
    else:
        return np.diag(parity_diag)


# Matrix inversion


def _max_abs(matrix):
    if scipy.sparse.issparse(matrix):
        return 0.0 if matrix.nnz == 0 else float(np.max(np.abs(matrix.data)))
    return float(np.max(np.abs(matrix)))


@cache
def _boson_diagonals(truncation):
    return tuple(
        (
            offset,
            tuple(
                1 / sqrt(prod(range(m + 1, m + abs(offset) + 1)))
                for m in range(truncation + 1 - abs(offset))
            ),
        )
        for offset in range(-truncation, truncation + 1)
    )


def _boson_coefficients(matrix, truncation, threshold=1e-10):
    """Decompose a bosonic matrix into normal-ordered coefficient blocks.

    Parameters
    ----------
    matrix : np.ndarray | scipy.sparse.csr_array
        Matrix matching the normalized mode cutoffs.
    truncation : list[int]
        Cutoff for each bosonic mode.
    threshold : float
        Omit blocks whose maximum absolute entry is not larger than this value.

    Returns
    -------
    dict[tuple[tuple[int, int], ...], complex | float]
        Creation/annihilation powers mapped to scalar coefficients.
    """
    coefficients = {(): matrix}
    for cutoff in truncation:
        dimension = cutoff + 1
        next_coefficients = {}
        for prefix, block in coefficients.items():
            block_size = block.shape[0] // dimension
            for offset, scales in _boson_diagonals(cutoff):
                shift = abs(offset)
                values = []
                for m, scale in enumerate(scales):
                    row, column = (m + offset, m) if offset >= 0 else (m, m + shift)
                    values.append(
                        scale
                        * block[
                            row * block_size : (row + 1) * block_size,
                            column * block_size : (column + 1) * block_size,
                        ]
                    )

                for power in range(len(values)):
                    coefficient = values[0] / factorial(power)
                    p, q = (
                        (power + shift, power)
                        if offset >= 0
                        else (power, power + shift)
                    )
                    if _max_abs(coefficient) > threshold:
                        next_coefficients[prefix + ((p, q),)] = coefficient
                    values = [right - left for left, right in pairwise(values)]
        coefficients = next_coefficients

    return {powers: coefficient[0, 0] for powers, coefficient in coefficients.items()}


def BosonDecomposition(
    matrix: np.ndarray | scipy.sparse.csr_array,
    truncation: list[int] | int,
    threshold: float = 1e-10,
) -> dict[tuple[tuple[int, int], ...], complex | float]:
    """Decompose a matrix into normal-ordered bosonic coefficients.

    Parameters
    ----------
    matrix : np.ndarray | scipy.sparse.csr_array
        Square matrix with dimension ``prod(t + 1 for t in truncation)``.
    truncation : list[int] | int
        Per-mode cutoffs; a scalar describes one mode.
    threshold : float
        Omit coefficients whose absolute value is not larger than this value.

    Returns
    -------
    dict[tuple[tuple[int, int], ...], complex | float]
        Per-mode ``(creation, annihilation)`` powers mapped to coefficients.

    Raises
    ------
    ValueError
        If the matrix shape and truncation are incompatible.
    """

    truncation = [truncation] if isinstance(truncation, int) else truncation
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix is not square.")
    if matrix.shape[0] != prod(t + 1 for t in truncation):
        raise ValueError("Matrix dimension is incompatible with truncation.")
    return _boson_coefficients(matrix, truncation, threshold)


def _boson_monomial(powers, operators):
    return prod(
        Dagger(operator) ** p * operator**q
        for (p, q), operator in zip(powers, operators)
    )


def _symbolic_coefficient(coefficient):
    """Keep exact expressions; rationalize numerical values to relative 1e-12."""
    if isinstance(coefficient, sympy.Basic):
        return coefficient
    return sympy.nsimplify(
        coefficient, rational=True, tolerance=1e-12 * abs(coefficient)
    )


def boson_to_operators(
    matrix: np.ndarray | scipy.sparse.csr_array,
    operators: list[sympy.Expr],
    truncation: list[int] | int,
    threshold: float = 1e-10,
) -> sympy.Expr:
    """Recover a normal-ordered bosonic expression from a matrix.

    Parameters
    ----------
    matrix : np.ndarray | scipy.sparse.csr_array
        Square matrix in the truncated bosonic basis.
    operators : list[sympy.Expr]
        Bosonic modes in tensor-product order.
    truncation : list[int] | int
        Per-mode cutoffs; a scalar applies to every mode.
    threshold : float
        Omit coefficients whose absolute value is not larger than this value.

    Returns
    -------
    sympy.Expr
        Normal-ordered expression in ``operators``.
    """

    truncation = _truncation_list(truncation, len(operators))
    return sympy.expand(
        sum(
            _symbolic_coefficient(coefficient) * _boson_monomial(powers, operators)
            for powers, coefficient in BosonDecomposition(
                matrix, truncation, threshold
            ).items()
        )
    )
