from .backends import _FermionDataSparseCOO, _FermionDataSparseCSR, _FermionDataArray
from .fock_basis import FockBasis
import numpy as np
from typing import List


class FermionData:
    """
    User facing class, storing the data representation of a FermionSequence in a given basis.
    #Note: perhaps somewhat convoluted to have this extra composite class structure? Motivation was to have a single user facing class but specific behavior for
            the different datastructures, rather than having a catch-all class. This allows readily extending to add different datastructure if desired.
    """

    _backends_dict = {
        "coo": _FermionDataSparseCOO,
        "csr": _FermionDataSparseCSR,
        "array": _FermionDataArray,
    }

    def __init__(self, fock_basis, data, data_backend: str = "coo"):
        self._backend = self._backends_dict.get(data_backend)(fock_basis, data)

    def _repr_markdown_(self):
        n_elements = len(self._backend)
        dim = self._backend.dim
        dtype = np.asarray(self._backend.values).dtype

        md = []
        md.append(
            f"< FermionSequence data representation of type `{dtype}`  \n"
            f"with {n_elements} stored elements and dimension {dim}x{dim}  >"
        )
        return "\n".join(md)

    def get(self):
        return self._backend.get()

    def __call__(self, value_dict):
        sympy_metadata = getattr(self._backend, "sympy_metadata", None)
        if sympy_metadata is None or len(sympy_metadata["param_dict"]) == 0:
            raise AttributeError(
                "This FermionData object does not contain symbolic parameter metadata. "
                "Compile a FermionSequence with SymPy-prefactors before calling update_values."
            )

        affected_term_indices = set()
        for sympy_key, new_value in value_dict.items():
            if sympy_key not in sympy_metadata["param_dict"]:
                raise KeyError(
                    f"SymPy key {sympy_key} not found in the compiled symbolic parameter metadata."
                )
            sympy_metadata["param_values"][sympy_key] = new_value
            affected_term_indices.update(sympy_metadata["param_dict"][sympy_key])

        self._backend.update_values(
            affected_term_indices, sympy_metadata["param_values"]
        )

    update_values = __call__  ## Alias

    def connected_components(self) -> List[List[int]]:
        """
        Calculate the connected components of the FockStates represented by this OperSequenceData.
        """
        rows = self._backend.rows
        cols = self._backend.cols

        if len(rows) == 0:
            return []

        components = [[rows[0], cols[0]]]
        for r, c in zip(rows[1:], cols[1:]):
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

    def to_block_diagonal_basis(self) -> FockBasis:
        """
        Calculate a new order for the basis such that the system is block diagonal
        """

        components = self.connected_components()
        basis_states = list(self._backend.fock_basis.states)
        n_states = len(basis_states)

        if len(components) == 0:
            return FockBasis(basis_states, N=self._backend.fock_basis.N)

        covered_indices = set()
        new_states_order = []

        for component in components:
            unique_component = []
            for idx in component:
                idx = int(idx)
                if idx not in covered_indices:
                    covered_indices.add(idx)
                    unique_component.append(idx)
            new_states_order.extend(unique_component)

        isolated_indices = [
            idx for idx in range(n_states) if idx not in covered_indices
        ]
        new_states_order.extend(isolated_indices)

        states_block_basis = [basis_states[i] for i in new_states_order]
        return FockBasis(states_block_basis)
