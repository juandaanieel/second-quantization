# codespell:ignore COO
from scipy.sparse import coo_array, csr_array
import numpy as np

from typing import List


class FermionSequenceData:
    """
    Stores data representing the action of an OperSequence instance on a specific
    Maintains the minimal data in sparse format (rows, columns, values), with additional
    information that allows for quickly updating values when weights of the OperSequence are adjusted.

    Implements conversion to numpy arrays (.to_array()) or scipy coo_matrix format (.to_sparse_coo())

    Attrs:
        fock_basis (FockStates): the FockStates instance for which the data was generated.
        rows (np.ndarray[int]): row indices of non-zero elements
        cols (np.ndarray[int]): column indices of non-zero elements
        values (np.ndarray[complex]): values of non-zero elements
        parities (np.ndarray[int]): the sign under action of operators giving rise to a non-zero element
        type_dict (dict): dict encoding which operators gave rise to elements in rows/cols
        data_array (Optional np.ndarray[complex]): full numpy array generated from rows,cols and values
    """

    def __init__(self, sparse_data: np.ndarray, fock_state_size: int):
        self.N = fock_state_size
        self.rows = sparse_data[0]
        self.cols = sparse_data[1]
        self.parities = sparse_data[2]
        self.type_dict = sparse_data[3]
        self.values = sparse_data[4]

    def _repr_markdown_(self) -> str:
        print("Rows: ", self.rows)
        print("Cols: ", self.cols)
        print("Vals: ", self.values)
        return ""

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

        ## If it has been created, update the matrix
        if hasattr(self, "data_array"):
            for key in type_keys:
                type_slice = self.type_dict[key]
                rows = self.rows[type_slice]
                cols = self.cols[type_slice]
                signs = self.parities[type_slice]
                old_values = signs * old_val
                filter_diagonal = np.where(
                    rows != cols
                )  ## Prevent double counting of the diagonal terms
                np.add.at(
                    self.data_array,
                    (
                        np.append(rows, cols[filter_diagonal]),
                        np.append(cols, rows[filter_diagonal]),
                    ),
                    np.append(
                        -old_values + new_values,
                        -np.conj(old_values[filter_diagonal])
                        + np.conj(new_values[filter_diagonal]),
                    ),
                )

    def to_array(self) -> np.ndarray:
        """
        Returns:

        """
        if hasattr(self, "data_array"):
            return self.data_array
        array_size = 4**self.N

        arr = np.zeros((array_size, array_size), dtype=complex)

        filter_diagonal = self.rows != self.cols
        rows = np.concatenate((self.rows, self.cols[filter_diagonal]))
        cols = np.concatenate((self.cols, self.rows[filter_diagonal]))
        new_values = self.values * self.parities
        vals = np.concatenate((new_values, np.conj(new_values[filter_diagonal])))

        np.add.at(arr, (rows, cols), vals)
        self.data_array = arr
        return arr

    def to_sparse_coo(self) -> coo_array:
        """
        Converts the OperSequenceData to a sparse COO matrix format.
        Returns:
            coo_array: A sparse matrix in COO format.
        """
        N = 4**self.N
        filter_diagonal = np.where(self.rows != self.cols)
        rows = np.append(self.rows, self.cols[filter_diagonal])
        cols = np.append(self.cols, self.rows[filter_diagonal])
        vals = np.append(
            self.values * self.parities,
            np.conj(self.values[filter_diagonal]) * self.parities[filter_diagonal],
        )
        return coo_array((vals, (rows, cols)), shape=(N, N))

    def to_sparse_csr(self) -> csr_array:
        """
        Converts the OperSequenceData to a sparse CSR matrix format.
        Returns:
            csr_array: A sparse matrix in CSR format.
        """
        N = 4**self.N
        filter_diagonal = np.where(self.rows != self.cols)
        rows = np.append(self.rows, self.cols[filter_diagonal])
        cols = np.append(self.cols, self.rows[filter_diagonal])
        vals = np.append(
            self.values * self.parities,
            np.conj(self.values[filter_diagonal]) * self.parities[filter_diagonal],
        )
        return csr_array((vals, (rows, cols)), shape=(N, N))

    def connected_components(self) -> List[List[int]]:
        """
        Calculate the connected components of the FockStates represented by this OperSequenceData.
        """
        idx = [[0]]
        components = [[self.rows[0], self.cols[0]]]
        for r, c in zip(self.rows[1:], self.cols[1:]):
            present_in = []
            for idx, comp in enumerate(components):
                r_present = r in comp
                c_present = c in comp
                if r_present and c_present:
                    present_in.append(idx)
                elif r_present:
                    components[idx].extend([c])
                    present_in.append(idx)
                elif c_present:
                    components[idx].extend([r])
                    present_in.append(idx)
            if len(present_in) == 0:
                components.append([r, c])
            elif len(present_in) > 1:
                for p_idx in present_in[-1:0:-1]:
                    components[present_in[0]].extend(components[p_idx])
                    components.pop(p_idx)

        connected_components = [list(set(comp)) for comp in components]
        return connected_components

    """
    def to_block_diagonal_basis(self) -> FockStates:
        ###
        ###Calculate a new order for the basis such that the system is block diagonal
        ###
        components = self.connected_components()
        reverse_hash = []
        for key in self.fock_basis.hashed.keys():
            reverse_hash.append(key)

        new_states_order = []
        for j in range(len(components)):
            new_states_order.extend(list(components[j]))
        states_block_basis= [reverse_hash[i] for i in new_states_order]
        return FockStates(states_block_basis)
    """
