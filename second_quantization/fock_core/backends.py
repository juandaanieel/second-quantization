from .fock_basis import FockBasis
import numpy as np
from scipy.sparse import coo_array, csr_array


class _FermionDataBackend:
    """
    Common class handling/enforcing common properties for all data backends.
    """

    def __init__(self, fock_basis: "FockBasis"):
        self.fock_basis = fock_basis
        self.dim = len(fock_basis)

    def update_values():
        raise NotImplementedError

    def get():
        raise NotImplementedError

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
        Converts FermionData to scipy sparse COO matrix format.
        Returns:
            coo_array: A sparse matrix in scipy COO format.
        """
        return coo_array(
            (self.values, (self.rows, self.cols)), shape=(self.dim, self.dim)
        )

    def to_array(self) -> np.ndarray:
        """
        Converts FermionData to a numpy ndarray
        Returns:
            np_array: numpy ndarray
        """
        array_size = self.dim
        np_array = np.zeros((array_size, array_size), dtype=complex)
        np.add.at(np_array, (self.rows, self.cols), self.values)
        return np_array

    def __len__(self):
        return len(self.values)


class _FermionDataSparseCSR(_FermionDataBackend):
    def __init__(self, fock_basis: FockBasis, sparse_data: dict):
        super().__init__(fock_basis)


class _FermionDataArray(_FermionDataBackend):
    def __init__(self, fock_basis: FockBasis, dense_data: tuple):
        super().__init__(fock_basis)
        self.rows = dense_data[0]
        self.cols = dense_data[1]
        self.values = dense_data[2]
        self.parities = dense_data[3]
        self.type_dict = dense_data[4]
        self.data_array = self.to_array()

    def update_values(
        self, type_keys: list[str], old_val: complex, new_val: complex
    ) -> None:
        for key in type_keys:
            type_slice = self.type_dict[key]
            rows = self.rows[type_slice]
            cols = self.cols[type_slice]
            signs = self.parities[type_slice]

            np.add.at(
                self.data_array,
                (rows, cols),
                signs * (new_val - old_val),
            )

    def get(self) -> coo_array:
        """
        Returns the native data-type format
        """
        return self.data_array


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
        fock_basis: FockBasis,
        sparse_data: tuple,
    ):
        super().__init__(fock_basis)
        self.rows = sparse_data[0]
        self.cols = sparse_data[1]
        self.values = sparse_data[2]
        self.parities = sparse_data[3]
        self.type_dict = sparse_data[4]

    def update_values(
        self, type_keys: list[str], old_val: complex, new_val: complex
    ) -> None:
        ## Set the new value for the relevant positions
        for key in type_keys:
            type_slice = self.type_dict[key]
            signs = self.parities[type_slice]
            new_values = signs * new_val
            self.values[type_slice] = new_values

    def get(self) -> coo_array:
        """
        Returns the native data-type format
        """
        return self.to_sparse_coo()
