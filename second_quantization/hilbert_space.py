from itertools import product
import math
from sympy.physics.quantum import Dagger
from sympy.core.singleton import S
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.boson import BosonOp
import sympy
import numpy as np
import scipy
import inspect
from sympy.physics.quantum.operatorordering import normal_ordered_form
from typing import List
from . import fermions, bosons


def _split_operators(operators):
    fermion_ops, boson_ops = [], []
    for operator in operators:
        if isinstance(operator, FermionOp):
            fermion_ops.append(operator)
        elif isinstance(operator, BosonOp):
            boson_ops.append(operator)
        else:
            raise ValueError("Can't submit non operator types")
    return fermion_ops, boson_ops


def basis_operators(operators, sparse, truncation=None):
    """Split fermionic and bosonic operators and build their matrix bases.

    Fermionic operators are mapped through ``fermion_basis``; bosonic operators
    are mapped through ``boson_basis`` with the provided truncation. The return
    value is a pair of dictionaries whose keys are ``1`` and the operator
    symbols/powers that are explicitly constructed.

    Raises
    ------
    ValueError
        If non-fermion/boson symbols are provided or bosonic truncation is
        omitted when bosons are present.
    """
    fermion_ops, boson_ops = _split_operators(operators)

    fermion_dict = (
        fermions.fermion_basis(fermion_ops, sparse)
        if fermion_ops
        else {1: np.array([1])}
    )
    boson_dict = (
        bosons.boson_basis(boson_ops, truncation, sparse)
        if boson_ops
        else {1: np.array([1])}
    )
    return fermion_dict, boson_dict


def parity_operator(
    operators: list[sympy.Expr], sparse: bool, truncation: int | list[int] | None = None
):
    """Construct total occupation parity: 0 for even and 1 for odd.

    Fermion and boson occupations contribute modulo two. Tensor factors follow
    the order fermions then bosons, as in ``to_matrix``.
    """

    fermion_ops, boson_ops = _split_operators(operators)

    f_parity = (
        fermions.fermion_parity(fermion_ops, sparse)
        if fermion_ops
        else np.zeros((1, 1), dtype=int)
    )
    b_parity = (
        bosons.boson_parity(boson_ops, truncation, sparse)
        if boson_ops
        else np.zeros((1, 1), dtype=int)
    )

    parity_diag = np.bitwise_xor.outer(f_parity.diagonal(), b_parity.diagonal()).ravel()
    if sparse:
        return scipy.sparse.diags(parity_diag, format="csr", dtype=parity_diag.dtype)
    else:
        return np.diag(parity_diag)


def to_matrix(
    expression: sympy.Expr,
    operators: list[
        sympy.physics.quantum.fermion.FermionOp | sympy.physics.quantum.boson.BosonOp
    ],
    sparse: bool,
    operator_dict: dict = None,
    truncation: int | list[int] = None,
) -> dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]:
    """Convert a symbolic operator expression to matrices.

    Expands ``expression`` into normal products of the provided fermionic and/or
    bosonic operators, builds matrix representations via tensor products, and
    groups terms by their purely symbolic prefactor. Creation operators are
    inferred via ``Dagger`` and implemented as conjugate-transposed annihilation
    matrices.

    Args:
        expression: SymPy expression containing the operators to expand.
        operators: Modes whose stable fermion partition, followed by their stable
            boson partition, sets the tensor-product ordering.
        sparse: If True, use sparse matrices for all operations.
        operator_dict: Optional cached ``(fermion_dict, boson_dict)`` from
            ``basis_operators`` to avoid recomputation.
        truncation: Truncation passed to bosonic basis construction when needed.

    Returns:
        Dict mapping symbolic prefactors to the summed matrix representation of
        all terms that share that prefactor.
    """

    if operator_dict is None:
        fermion_dict, boson_dict = basis_operators(operators, sparse, truncation)
    else:
        fermion_dict, boson_dict = operator_dict

    if sparse:

        def kron(a, b):
            return scipy.sparse.kron(a, b, format="csr")
    else:
        kron = np.kron

    dict_matrices = {}
    # Expand the expression to ensure all terms are separated
    expression = expression.expand()
    # Iterate over each term in the expression
    for term, coeff in expression.as_coefficients_dict().items():
        term_mat = float(coeff)
        symbol = S.One

        fermion_factors = fermion_dict[1].copy()
        boson_factors = boson_dict[1].copy()

        for factor in term.as_ordered_factors():
            base, power = factor.as_base_exp()
            if factor in fermion_dict:
                fermion_factors = fermion_factors @ fermion_dict[factor]
            elif Dagger(factor) in fermion_dict:
                fermion_factors = (
                    fermion_factors @ fermion_dict[Dagger(factor)].conj().T
                )
            elif factor in boson_dict:
                boson_factors = boson_factors @ boson_dict[factor]
            elif Dagger(factor) in boson_dict:
                boson_factors = boson_factors @ boson_dict[Dagger(factor)].conj().T
            elif (
                isinstance(base, BosonOp)
                and (base in operators or Dagger(base) in operators)
                and power.is_Integer
                and power > 0
            ):
                # Powers beyond the truncated bosonic basis vanish.
                boson_factors = 0 * boson_factors
            else:
                symbol *= factor

        term_mat = term_mat * kron(fermion_factors, boson_factors)

        if symbol in dict_matrices.keys():
            dict_matrices[symbol] += term_mat
        else:
            dict_matrices[symbol] = term_mat
    return dict_matrices


def make_dict_callable(
    hamiltonian_dict: dict[sympy.Expr, np.ndarray],
) -> callable:
    """Create a callable function from a dictionary of SymPy expressions and NumPy arrays.

    This function takes a dictionary where keys are symbolic expressions (containing
    free symbols) and values are NumPy arrays, and creates a callable function that
    evaluates the weighted sum of the arrays based on the symbolic expressions.

    Args:
        hamiltonian_dict: A dictionary where keys are SymPy expressions (which may
            contain free symbols) and values are NumPy arrays of the same shape.

    Returns:
        A callable function that takes values for all free symbols found in the
        dictionary keys and returns the weighted sum: Σ(expr_value * array) where
        expr_value is the numerical evaluation of each symbolic expression.

    Raises:
        ValueError: If any symbol names would be converted to Dummy variables by
            SymPy's lambdify function. This typically happens with special characters
            or reserved names.

    Example:
        ```python
        import sympy as sp
        import numpy as np

        # Define symbolic parameters
        x, y = sp.symbols('x y')

        # Create dictionary with symbolic expressions and matrices
        ham_dict = {
            x: np.array([[1, 0], [0, 0]]),
            y: np.array([[0, 1], [1, 0]])
        }

        # Create callable function
        func = make_dict_callable(ham_dict)

        # Evaluate at specific parameter values
        result = func(x=2.0, y=1.5)  # Returns 2.0 * first_matrix + 1.5 * second_matrix
        ```
    """
    all_symbols = list(
        sympy.ordered(sum([k for k in hamiltonian_dict.keys()]).free_symbols)
    )

    array_of_expr = sympy.Array(list(hamiltonian_dict.keys()))
    callable_expr = sympy.lambdify(all_symbols, array_of_expr, modules="numpy")
    tensor = np.array([v for v in list(hamiltonian_dict.values())])
    sum_rule = [None for _ in range(len(tensor.shape) - 1)]

    # check if the arguments of the callable contain dummys
    arg_names = list(inspect.signature(callable_expr).parameters.keys())
    if np.any(["Dummy" in name for name in arg_names]):
        raise ValueError(
            "Variable names must be chosen such that they are not lambdified to Dummy variables. Avoid special characters"
        )

    def func(*args, **kwargs):
        prefac = callable_expr(*args, **kwargs)[:, *sum_rule]
        return np.sum(prefac * tensor, axis=0)

    # adjust signature
    func.__signature__ = inspect.signature(callable_expr)

    return func


def _generate_all_powers(
    operators: list[sympy.Expr],
    truncation: int | list[int],
    boson_list: list[sympy.Expr] | None = None,
):
    boson_list = boson_list or [
        op for op in operators if isinstance(op, sympy.physics.quantum.boson.BosonOp)
    ]
    exponents = []
    for op in operators:
        if isinstance(op, FermionOp):
            exponents.append(np.array([0, 1], dtype=int))
        elif isinstance(op, BosonOp):
            if isinstance(truncation, int):
                tr = truncation
            else:
                id = boson_list.index(op)
                tr = truncation[id]
            exponents.append(np.arange(0, tr + 1, dtype=int))
    return np.fromiter(product(*exponents), dtype=np.dtype((int, len(operators))))


def _infer_boson_truncation(
    operators: list[sympy.Expr], operator_dict: dict | tuple | None
) -> list[int]:
    if operator_dict is None:
        raise ValueError("truncation is required for bosonic operators")
    boson_dict = operator_dict[1] if isinstance(operator_dict, tuple) else operator_dict

    truncations = []
    for op in operators:
        if not isinstance(op, BosonOp):
            continue
        max_power = None
        for key in boson_dict.keys():
            if key == op:
                max_power = 1 if max_power is None else max(max_power, 1)
            elif isinstance(key, sympy.Pow) and key.base == op:
                try:
                    exp = int(key.exp)
                except (TypeError, ValueError):
                    continue
                max_power = exp if max_power is None else max(max_power, exp)
        if max_power is None:
            raise ValueError(f"Unable to infer truncation for boson operator {op}")
        truncations.append(max_power)
    return truncations


def symbolic_basis(
    operators: list[FermionOp | BosonOp], truncation: int | list[int] | None = None
) -> list[sympy.Expr]:
    """Generate creation monomials in matrix basis order.

    Fermions precede bosons, preserving the order within each particle type,
    as in ``to_matrix``. The rightmost mode varies fastest.

    Args:
        operators: Fermionic and/or bosonic modes.
        truncation: Maximum bosonic occupation, either shared or per boson.
            Required only when bosons are present.

    Returns:
        Creation monomials acting on vacuum, starting with ``sympy.S.One``.
        Bosonic powers are unnormalized: occupation ``n`` contributes a factor
        of ``sqrt(n!)`` when applied to vacuum.

    Example:
        For 2 fermions `[c, d]`, returns:
        - `[1, d†, c†, c†*d†]` representing vacuum, single occupations, and double occupation.
    """
    fermion_ops, boson_ops = _split_operators(operators)
    operators = fermion_ops + boson_ops
    truncation = bosons._truncation_list(truncation, len(boson_ops))
    symbols = []
    strings = _generate_all_powers(operators, truncation)

    for string in strings:
        sub_symbol = sympy.S.One
        for idx, power in enumerate(string):
            if power == 1:
                sub_symbol *= Dagger(operators[idx])
            elif power > 1:
                sub_symbol *= Dagger(operators[idx]) ** power
        symbols.append(sub_symbol)
    return symbols


def _get_all_combinations(
    subset: List[sympy.Expr],
    all_operators: List[sympy.Expr],
    sparse: bool,
    operator_dict: dict = None,
    truncation: int | List[int] | None = None,
    return_powers: bool = False,
) -> dict[sympy.Expr : np.ndarray] | tuple[dict[sympy.Expr, np.ndarray], np.ndarray]:
    """Build matrices for all binary products of a fermionic subset.

    Enumerates ``2**len(subset)`` occupation patterns (presence/absence) for the
    provided fermionic operators, converts each symbolic product to a matrix via
    ``to_matrix``, and returns a dictionary keyed by the symbolic product.

    Args:
        subset: Fermionic operators to include in the combinations.
        all_operators: Full ordered operator list used for ``to_matrix``.
        sparse: Whether to keep intermediate matrices sparse.
        operator_dict: Optional cached operator dictionaries.
    """
    subset = [subset] if not isinstance(subset, list) else subset
    operator_list = {}
    symbolic_list = []
    boson_list = [op for op in all_operators if isinstance(op, BosonOp)]
    if truncation is None:
        truncation = (
            _infer_boson_truncation(all_operators, operator_dict) if boson_list else 1
        )
    all_combinations = _generate_all_powers(subset, truncation, boson_list)

    for combination in all_combinations:
        symbolic_list.append(
            np.prod([b_op**bit for b_op, bit in zip(subset, combination)][::-1])
        )

    for symbol in symbolic_list:
        operator_list[symbol] = to_matrix(
            symbol,
            operators=all_operators,
            sparse=sparse,
            operator_dict=operator_dict,
            truncation=truncation,
        )[1]
    if return_powers:
        return operator_list, all_combinations
    return operator_list


def partial_trace_generators(
    subset: list[sympy.Expr],
    all_operators: list[sympy.Expr],
    sparse: bool,
    operator_dict: dict = None,
    truncation: int | list[int] | None = None,
) -> np.ndarray:
    """Projectors for tracing out fermionic and/or bosonic modes.

    Constructs vectors that project onto each configuration of the complement of
    ``subset`` while fixing ``subset`` in vacuum, enabling partial traces via the
    recipe documented in the return value description.

    Args:
        subset: Modes to remove via tracing.
        all_operators: All modes in the system, ordered as in ``to_matrix``.
        sparse: Whether to keep intermediate matrices sparse.
        operator_dict: Optional cached operator dictionaries.
        truncation: Maximum bosonic occupations, or inferred from ``operator_dict``.

    Returns:
        Array of projectors ``p`` such that ``sum(p[:, i, :] @ M @ p[:, i, :].conj().T)``
        traces out ``subset`` from matrix ``M``. Remaining modes follow the
        fermions-first order of ``symbolic_basis``.
    """
    fermion_ops, boson_ops = _split_operators(all_operators)
    all_operators = fermion_ops + boson_ops
    set_subset = set([subset] if not isinstance(subset, list) else subset)
    complement = [element for element in all_operators if element not in set_subset]

    subset_operators, subset_powers = _get_all_combinations(
        subset=subset,
        all_operators=all_operators,
        sparse=sparse,
        operator_dict=operator_dict,
        truncation=truncation,
        return_powers=True,
    )
    complement_operators, complement_powers = _get_all_combinations(
        subset=complement,
        all_operators=all_operators,
        sparse=sparse,
        operator_dict=operator_dict,
        truncation=truncation,
        return_powers=True,
    )

    size = list(subset_operators.values())[0].shape[0]
    vacuum = np.zeros(size)
    vacuum[0] = 1

    vecs = []

    def _boson_norm(ops, powers):
        norm = 1.0
        for op, power in zip(ops, powers):
            if isinstance(op, BosonOp):
                norm *= math.sqrt(math.factorial(int(power)))
        return norm

    subset_items = list(subset_operators.values())
    subset_scaled = [
        mat / _boson_norm(subset, powers)
        for mat, powers in zip(subset_items, subset_powers)
    ]

    complement_items = list(complement_operators.values())
    complement_scaled = [
        mat / _boson_norm(complement, powers)
        for mat, powers in zip(complement_items, complement_powers)
    ]

    for element in complement_scaled:
        aux = []
        for g in subset_scaled:
            generator = g @ element
            aux.append(vacuum @ generator)
        vecs.append(aux)

    return np.array(vecs)


def to_operators(
    matrix: dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]
    | np.ndarray
    | scipy.sparse.csr_array,
    basis: list[sympy.Expr],
    truncation: int | list[int] | None = None,
) -> sympy.Expr:
    """Recover a normal-ordered operator expression from a matrix.

    Dispatches to the appropriate inversion backend based on the operator
    types present in ``basis``:

    * **Fermionic only** – Pauli decomposition + Jordan–Wigner inversion
      (original behaviour, requires dim = 2^n).
    * **Bosonic only** – diagonal falling-factorial decomposition with
      finite-difference inversion (requires ``truncation``).
    * **Mixed** – Pauli decomposition on the fermionic tensor structure
      followed by bosonic inversion on each coefficient block (requires
      ``truncation``).

    If a dictionary of matrices is supplied, each entry is converted and
    scaled by its symbolic key before summing.

    Parameters
    ----------
    matrix : ndarray, csr_array, or dict thereof
        Square matrix (or dict of matrices) in the Hilbert space defined by
        ``basis`` and the tensor-product ordering of :func:`to_matrix`.
    basis : list of FermionOp / BosonOp
        Operators defining the Hilbert space. They are stably partitioned into
        fermions followed by bosons, matching :func:`to_matrix`.
    truncation : int, list of int, or None
        Required when ``basis`` contains bosonic operators; passed to the
        bosonic inversion backend.

    Returns
    -------
    sympy.Expr
        Normal-ordered symbolic expression.
    """
    fermion_ops, boson_ops = _split_operators(basis)
    truncation = bosons._truncation_list(truncation, len(boson_ops))
    dimension = 2 ** len(fermion_ops) * math.prod(t + 1 for t in truncation)
    matrices = matrix.items() if isinstance(matrix, dict) else ((S.One, matrix),)
    fermion_cache = {"": S.One}
    terms = {}

    for prefactor, value in matrices:
        if value.shape != (dimension, dimension):
            raise ValueError("Matrix dimension is incompatible with basis.")
        for pauli_string, block in fermions._pauli_coefficients(
            value, len(fermion_ops)
        ).items():
            if pauli_string not in fermion_cache:
                fermion_cache[pauli_string] = fermions.string_to_fermion_operators(
                    pauli_string, fermion_ops
                )
            coefficients = bosons._boson_coefficients(
                block, truncation, 1e-10 if boson_ops else 0.0
            )
            for powers, coefficient in coefficients.items():
                coefficient = bosons._symbolic_coefficient(coefficient)
                terms[powers] = terms.get(powers, S.Zero) + (
                    prefactor * coefficient * fermion_cache[pauli_string]
                )

    result = S.Zero
    for powers, fermion_expr in terms.items():
        if fermion_ops:
            fermion_expr = normal_ordered_form(
                sympy.expand(fermion_expr), independent=True
            )
        result += fermion_expr * bosons._boson_monomial(powers, boson_ops)
    return sympy.expand(result)
