from sympy.physics.quantum import Dagger
from sympy.core.singleton import S
import sympy
from functools import reduce
import numpy as np
import scipy
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

    parity_operator = reduce(
        kron,
        [sigma_z] * size,
    )  # Parity operator is the product of all sigma_z operators

    return basis_operators, parity_operator


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


def _get_all_combinations(
    subset: list[sympy.Expr], all_operators: list[sympy.Expr], sparse: bool
) -> dict[sympy.Expr : np.ndarray]:
    """
    Get symbolic representation of all possible combinations of operators of the input list

    Parameters
    ----------
        subset list[sympy.Expr]:
            Subset of of operators one wants the combinations of
        all_operators list[sympy.Expr]:
            A list of sympy expressions representing the fermionic operators of the system.
        sparse: bool:
            Bool to set whether dense numpy arrays or sparse scipy arrays are used
    Returns
    -------
        dict[sympy.Expr, np.ndarray]:
            Dict with all combinations of operators within the subset as keys and their corresponding matrix
            representation as value
    """
    subset = [subset] if not isinstance(subset, list) else subset
    N_max = 2 ** len(subset)
    n_ops = len(subset)
    operator_list = {}
    symbolic_list = []

    for i in range(N_max + 1):
        bit_string = [
            0 for _ in range(n_ops - len(list(map(int, format(i, "b")))))
        ] + list(map(int, format(i, "b")))
        symbolic_list.append(
            np.prod([b_op**bit for b_op, bit in zip(subset, bit_string)])
        )

    for symbol in symbolic_list:
        operator_list[symbol] = to_matrix(
            symbol, operators=all_operators, sparse=sparse
        )[1]
    return operator_list


def partial_trace_generators(
    subset: list[sympy.Expr], all_operators: list[sympy.Expr], sparse: bool
) -> np.ndarray:
    """
    Generate the projectors required to compute the partial trace of a subset of operators

    Parameters
    ----------
        subset list[sympy.Expr]:
            Subset of operators to trace out
        all_operators list[sympy.Expr]:
            A list of sympy expressions representing the fermionic operators of the system.
        sparse: bool:
            Bool to set whether dense numpy arrays or sparse scipy arrays are used
    Returns
    -------
        np.ndarray:
            numpy array with the projectors that trace out the subset of operators specified in the argument.
            The final trace is taken by projecting a matrix M as
            np.sum([p[:,i,:] @ M @ p[:,i,:].conj().T for i in range(p.shape[1])])
    """
    set_subset = set([subset] if not isinstance(subset, list) else subset)
    complement = [element for element in all_operators if element not in set_subset]

    subset_operators = _get_all_combinations(
        subset=subset, all_operators=all_operators, sparse=sparse
    )
    complement_operators = _get_all_combinations(
        subset=complement, all_operators=all_operators, sparse=sparse
    )

    size = list(subset_operators.values())[0].shape[0]
    vacuum = np.zeros(size)
    vacuum[0] = 1

    vecs = []

    for element in complement_operators.values():
        aux = []
        for g in subset_operators.values():
            generator = g @ element
            aux.append(vacuum @ generator)
        vecs.append(aux)

    return np.array(vecs)
