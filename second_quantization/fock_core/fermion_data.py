# codespell:ignore COO
from .fock_basis import FockBasis
from scipy.sparse import coo_array, csr_array
import numpy as np


class _FermionDataBackend:
    """
    Common class handling/enforcing common properties for all data backends
    """

    def __init__(self, fock_basis: "FockBasis"):
        self.fock_basis = fock_basis
        self.dim = len(fock_basis)


class _FermionDataSparseCOO(_FermionDataBackend):
    """
    Stores data representing the action of a FermionSequence instance on a specific basis
    Maintains the data in sparse COO format (rows, columns, values), with additional
    information that allows for quickly updating values when weights of the FermionSequence are adjusted.

    Allows conversion to numpy arrays (.to_array()) or scipy csr_matrix format (.to_sparse_csr())
    Args:
        sparse_data (tuple): a 5-tuple containing the data in sparse format (rows, columns, values, parities, type_dict).
        fock_space (FockBasis): FockBasis object for which the data was generated.
    Attrs:
        rows (np.ndarray[int]): row indices of non-zero elements
        cols (np.ndarray[int]): column indices of non-zero elements
        values (np.ndarray[complex]): values of non-zero elements
        parities (np.ndarray[int]): the fermionic sign under action of operators for each element, used for updating values when weights of the FermionSequence are adjusted.
        type_dict (dict): dict encoding which operators gave rise to elements in rows/cols
    """

    def __init__(
        self,
        fock_basis: "FockBasis",
        sparse_data: np.ndarray,
    ):
        super().__init__(fock_basis)
        self.rows = sparse_data[0]
        self.cols = sparse_data[1]
        self.values = sparse_data[2]
        self.parities = sparse_data[3]
        self.type_dict = sparse_data[4]

    def _repr_markdown_(self):
        nnz = len(self.values)
        dim = self.dim
        dtype = np.asarray(self.values).dtype

        md = []
        md.append(
            f"< FermionSequence sparse representation of type `{dtype}`  \n"
            f"with {nnz} stored elements and shape {dim}x{dim}  >"
        )
        return "\n".join(md)

    '''TODO: reintegrate the matvec/matmul for LinearOperator compatibility
    def get_sparse_func(self) -> Callable:
        """
        Return a function that can be used by scipy's LinearOperator class
        for finding the lowest eigenvalues without storing a full matrix.

        Returns:
            sparse_func -> a function such that passing a vector x to sparse_func(x)
                            implements the matrix vector product to return a vector v
                            This is equivalent to getting the matrix form M with .to_array()
                            and applying M @ x = v
        """

        filter_diag = np.where(self.rows != self.cols)
        values = self.values * self.parities
        sparse_func = partial(
            fo.sparse_matvec,
            rows=np.append(self.rows, self.cols[filter_diag]),
            cols=np.append(self.cols, self.rows[filter_diag]),
            vals=np.append(values, np.conj(values[filter_diag])),
        )
        return sparse_func

    def __matmul__(self, vector: np.ndarray) -> np.ndarray:
        return self.get_sparse_func()(vector)
    '''

    def update_values(
        self, type_keys: list[str], old_val: complex, new_val: complex
    ) -> None:
        ## Set the new value for the relevant positions
        for key in type_keys:
            type_slice = self.type_dict[key]
            signs = self.parities[type_slice]
            new_values = signs * new_val
            self.values[type_slice] = new_values

    def to_array(self) -> np.ndarray:
        """
        Converts FermionSequenceData to a numpy ndarray
        Returns:
            np_array: numpy ndarray
        """
        if hasattr(self, "data_array"):
            return self.data_array
        array_size = self.dim
        np_array = np.zeros((array_size, array_size), dtype=complex)
        np.add.at(np_array, (self.rows, self.cols), self.values)
        self.data_array = np_array
        return np_array

    def to_sparse_csr(self) -> csr_array:
        """
        Converts FermionSequenceData to a sparse CSR matrix format.
        Returns:
            csr_array: A sparse matrix in CSR format.

        Notes:
            Incurs significant overhead for larger systems.
        """
        return csr_array(
            (self.values, (self.rows, self.cols)), shape=(self.dim, self.dim)
        )

    def to_sparse_coo(self) -> coo_array:
        """
        Converts FermionSequenceData to a sparse COO matrix format.
        Returns:
            coo_array: A sparse matrix in COO format.
        """
        return coo_array(
            (self.values, (self.rows, self.cols)), shape=(self.dim, self.dim)
        )

    def get(self) -> coo_array:
        """
        Returns the native data-type format
        """
        return self.to_sparse_coo()


class FermionData:
    """
    User facing class, storing the data representation of a FermionSequence in a given basis.
    """

    _backends_dict = {
        "coo": _FermionDataSparseCOO,
        #    'csr': "_FermionDataSparseCSR",
        #    'array': "_FermionDataArray",
    }

    def __init__(self, fock_basis, data, data_backend: str = "coo"):
        self._backend = self._backends_dict.get(data_backend)(fock_basis, data)

    def get(self):
        return self._backend.get()

    def __call__(self, value_dict):
        if not hasattr(self, "_sympy_dict"):
            raise AttributeError(
                "The FermionSequence object does not have the necessary attributes to set values by calling. Make sure the object was initialized from a sympy expression or that the _sympy_dict attribute is properly set."
            )

        for sympy_key, new_value in value_dict.items():
            if sympy_key not in self._sympy_dict.keys():
                raise KeyError(
                    f"SymPy key {sympy_key} not found in the FermionSequence's internal dictionary. Make sure to only set values for keys that are already present in the dictionary."
                )
            old_value = self._sympy_dict[sympy_key][0]
            sympy_to_opers_str_list = self._sympy_dict[sympy_key][1]
            self._sympy_dict[sympy_key] = (new_value, sympy_to_opers_str_list)

            self._backend.update_values(sympy_to_opers_str_list, old_value, new_value)

    update_values = __call__  ## Alias
