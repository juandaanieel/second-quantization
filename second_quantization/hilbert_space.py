from sympy.physics.quantum import Dagger
from sympy.core.singleton import S
import sympy
from functools import reduce
import numpy as np
import scipy
import inspect
from sympy.physics.quantum.operatorordering import normal_ordered_form

from .pauli_strings import PauliDecomposition


sigma_0 = np.array([[1, 0], [0, 1]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_minus = np.array([[0, 1], [0, 0]])


def basis_operators(
    operators: list[sympy.Expr], sparse: bool
) -> dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array :]:
    """
    Transform from second quantized operators to matrix representation in the Fock-space.

    Parameters
    ----------
        operators (list[sympy.Expr]):
            A list of sympy expressions representing the operators. sparse (bool): A boolean flag indicating whether to use sparse matrices.
    Returns
    -------
        dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]:
            A dictionary where the keys are the input operators and the values are their corresponding matrix representations in the Fock-space.
    Note:
    - The `operators` list does not require its elements to be `FermionOp`.
    - The fermion ordering follows the ordering in the list.
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
    basis_operators = {}
    for i, operator in enumerate(operators):
        basis_operators[operator] = reduce(
            kron, [sigma_z] * i + [sigma_minus] + [eye(2 ** (size - i - 1))]
        )

    return basis_operators


def to_matrix(
    expression: sympy.Expr,
    operators: list[sympy.physics.quantum.fermion.FermionOp],
    sparse: bool,
) -> dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]:
    """
    Transforms a sympy expression containing fermionic operators to a dictionary
    with the matrix representation of each term.

    Parameters
    ----------
    expression : sympy.Expr
        A sympy expression containing fermionic operators.
    operators : list[sympy.physics.quantum.fermion.FermionOp]
        A list of fermionic operators.
    sparse : bool
        If True, the matrices will be in sparse format. Otherwise, they will be dense.

    Returns
    -------
    dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]
        A dictionary where the keys are sympy expressions and the values are the
        corresponding matrix representations in either dense or sparse format.
    """

    operator_dict = basis_operators(operators, sparse)
    dimension = 2 ** len(operators)

    if not sparse:
        eye = np.eye
    else:

        def eye(N):
            return scipy.sparse.eye(N, format="csr")

    dict_matrices = {}
    # Expand the expression to ensure all terms are separated
    expression = expression.expand()
    # Iterate over each term in the expression
    for term, coeff in expression.as_coefficients_dict().items():
        term_mat = float(coeff) * eye(dimension)
        symbol = S.One

        for factor in term.as_ordered_factors():
            if factor in operator_dict:
                term_mat = term_mat @ operator_dict[factor]
            elif Dagger(factor) in operator_dict:
                term_mat = term_mat @ operator_dict[Dagger(factor)].T.conj()
            else:
                symbol *= factor

        if symbol in dict_matrices.keys():
            dict_matrices[symbol] += term_mat
        else:
            dict_matrices[symbol] = term_mat
    return dict_matrices


def to_operators(
    matrix: dict[sympy.Expr, np.ndarray | scipy.sparse.csr_array]
    | np.ndarray
    | scipy.sparse.csr_array,
    basis: list[sympy.Expr],
) -> sympy.Expr:
    """
    Parameters
    ----------
        matrix:
            The matrix to be converted. It can be a dictionary with sympy expressions as keys and numpy arrays or scipy sparse arrays as values, or it can be a numpy array or a scipy sparse array directly.
        basis (list[sympy.Expr]):
            A list of sympy expressions representing the basis operators.
    Returns
    -------
        sympy.Expr:
            The normal ordered operator expression corresponding to the input matrix.
    """

    if isinstance(matrix, dict):
        return sum([k * to_operators(v, basis) for k, v in matrix.items()])

    pauli_string = PauliDecomposition(matrix)
    term = S.Zero

    for string, prefactor in pauli_string.items():
        _term = S.One * prefactor
        prefactor_exponent = S.One

        for key, op in zip(string, basis):
            if key == "X":
                _term *= prefactor_exponent
                _term *= Dagger(op) + op
            elif key == "Y":
                _term *= prefactor_exponent
                _term *= 1j * (Dagger(op) - op)
            elif key == "Z":
                _term *= -(2 * Dagger(op) * op - 1)

            prefactor_exponent *= -(2 * Dagger(op) * op - 1)

        term += _term

    return normal_ordered_form(sympy.nsimplify(term).expand(), independent=True)


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

    # check if the arguments of the callable contain dummys
    arg_names = list(inspect.signature(callable_expr).parameters.keys())
    if np.any(["Dummy" in name for name in arg_names]):
        raise ValueError(
            "Variable names must be chosen such that they are not lambdified to Dummy variables. Avoid special characters"
        )

    def func(*args, **kwargs):
        prefac = callable_expr(*args, **kwargs)[:, None, None]
        return np.sum(prefac * tensor, axis=0)

    # adjust signature
    func.__signature__ = inspect.signature(callable_expr)

    return func
