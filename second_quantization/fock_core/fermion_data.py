from .backends import _FermionDataSparseCOO, _FermionDataSparseCSR, _FermionDataArray
import numpy as np


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
