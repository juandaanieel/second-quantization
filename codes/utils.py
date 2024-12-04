import numpy as np
import mumps
import scipy


def evaluate_hamiltonian(params, terms):
    """
    Compute the Hamiltonian from `terms` for a set of parameters
    """

    example_matrix = list(terms.values())[0]
    n = example_matrix.shape[0]
    h = np.zeros((n, n), dtype=complex)
    
    for key, value in terms.items():
        h += value * params[key]
    
    return h


def sparse_diag(matrix, k, sigma, mumps=False, **kwargs):
    """Call scipy.sparse.linalg.eigsh with mumps support.

    See scipy.sparse.linalg.eigsh for documentation.
    """
    if mumps:
        class LuInv(scipy.sparse.linalg.LinearOperator):
            def __init__(self, A):
                inst = mumps.Context()
                inst.analyze(A, ordering='pord')
                inst.factor(A)
                self.solve = inst.solve
                scipy.sparse.linalg.LinearOperator.__init__(self, A.dtype, A.shape)
    
            def _matvec(self, x):
                return self.solve(x.astype(self.dtype))
    
        opinv = LuInv(matrix - sigma * scipy.sparse.identity(matrix.shape[0]))
    else:
        opinv = None
    return scipy.sparse.linalg.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)


def parity_spectrum(matrix, p_tot, k=None, sigma=None, mumps=False):
    """
    Get parity resolved spectrum of input Hamiltonian

    Paramters:
    matrix: np.ndarray or scipy.sparse type
            Matrix of which the spectrum should be found

    p_tot: np.ndarray or scipy.sparse type
            Matrix representation of the total parity operator
            Type needs to be consistent with with input matrix

    k: int (default None)
            number of eigenvalues(-vectors) to calculate when sparse

    sigma: float (default None)
            Float for shift invert of the sparse solver

    mumps: boolean
            Boolean to decide whether to use mumps for LU decomposition
    """
    if (scipy.sparse.issparse(matrix) and
        scipy.sparse.issparse(p_tot)):
        k = 1 if k is None else k
        sigma = 0 if sigma is None else sigma
        eigs, vecs = sparse_diag(matrix, k, sigma, mumps)
    else:
        eigs, vecs = scipy.linalg.eigh(matrix)
    p_vals = vecs.conj().T@ p_tot @vecs
    assert np.isclose(0, np.linalg.norm(p_vals - np.diag(np.diag(p_vals))))
    even_eigs, even_vecs = eigs[np.diag(p_vals) > 0], vecs[:, np.diag(p_vals) > 0]
    odd_eigs, odd_vecs = eigs[np.diag(p_vals) < 0], vecs[:, np.diag(p_vals) < 0]
    return even_eigs, even_vecs, odd_eigs, odd_vecs
