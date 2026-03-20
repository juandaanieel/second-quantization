from ..fock_basis import FockBasis
from ..fermion_data import FermionData

from typing import List, Self, Any
import numpy as np
import sympy

def common_prefix_length(a, b):
    """
    Return the length of the common prefix shared by two operator lists.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

class FermionDataCompilerMixin:
    """
    Mixin implementing compilation of fermionic operator sequences into data
    representations on a chosen Fock basis.

    Expects the host class to provide:
    - `self.oper_list`
    - `self.weights`
    - `self.oper_list_to_str(...)`
    - `self.act_oper(...)`
    - `self.act_oper_list(...)`
    - `self.sort_lexicographical(...)`
    - optionally `self._sympy_dict`
    """
    ####################################################
    #####      CONSTRUCT DATA REPRESENTATION       #####
    ####################################################
    _compile_formats_list = ["coo", "csr", "array"]

    def compile(self, fock_basis: [int, FockBasis] = None, data_format="coo"):
        if data_format not in self._compile_formats_list:
            raise ValueError(
                f"Selected invalid data_format. "
                f"Implemented data formats are: {self._compile_formats_list}"
            )

        ## If given an integer, generate a FockBasis object for later convenience
        if isinstance(fock_basis, int):
            fock_basis = FockBasis(fock_basis)

        ## If no specific FockBasis was given, infer the size of the fockspace from the Hamiltonian itself
        elif fock_basis is None:
            non_empty_subsequences = [sublist for sublist in self.oper_list if len(sublist)]
            inferred_N = (
                (max(max(sublist) for sublist in non_empty_subsequences) >> 1) + 1
                if len(non_empty_subsequences)
                else 0
            )
            fock_basis = FockBasis(inferred_N)

        ## Construct the sparse representation object in specified basis
        sparse_data = FermionData(
            fock_basis, self._construct_data_repr(fock_basis), data_backend=data_format
        )

        ## Copy sympy identifiers for quick updating of sparse data representation
        if hasattr(self, "_sympy_dict"):
            sparse_data._sympy_dict = self._sympy_dict

        return sparse_data

    def _construct_data_repr(self, fock_basis):
        if fock_basis.is_ordered:
            return self._construct_data_repr_ordered(fock_basis)
        return self._construct_data_repr_restricted(fock_basis)

    def _construct_data_repr_ordered(self, fock_basis: FockBasis):
        """
        Applies the stored hamiltonian to a set of ordered fock states, wrapped in FockBasis.

        Returns:
            rows (ndarray[int]): row indices of non-zero terms
            cols (ndarray[int]): column indices of non-zero terms
            vals (ndarray[complex]): values of each row/col pair
            pars (ndarray[int]): relative signs of operators, required for updating the arrays at later stage
            type (dict): a dict pointing to where in the row/col memory the values related to a particular operator are stored
        """
        ### Initialize data storage
        ## Pre-calculate the amount of data that will be generated
        dim_basis = fock_basis.N
        n_required_data = 0
        for oper in self.oper_list:
            k = np.unique(np.array(oper) >> 1).size
            n_required_data += 2 ** (dim_basis - k)

        ## Fill empty arrays
        rows = np.empty(n_required_data, dtype=np.int32)
        cols = np.empty(n_required_data, dtype=np.int32)
        signs = np.empty(n_required_data, dtype=np.int8)
        values = np.zeros(n_required_data, dtype=np.complex128)

        types_dict, data_idx = {}, 0

        ## Unpack reference to array and weights (saves time inside the loop for larger systems)
        oper_list, weights = self.__class__.sort_lexicographical(
            self.oper_list, self.weights
        )
        type_strings = [self.oper_list_to_str(h) for h in oper_list]
        states = fock_basis.states
        act_func = self._act_oper_list_cached
        numeric_weights = all(not isinstance(w, sympy.Basic) for w in weights)

        ## Test Caching
        ones = np.ones(len(states), dtype=np.int8)
        data_cache = {"data": (states, states, ones)}
        cache_until_list = []
        for idx in range(len(oper_list) - 1):
            cache_until_list.append(
                common_prefix_length(oper_list[idx], oper_list[idx + 1])
            )
        cache_until_list.append(0)
        cached_until = 0

        ## Loop over operators and type string, apply each operator list to set of states
        for h, type_str, weight, cache_until in zip(
            oper_list, type_strings, weights, cache_until_list
        ):
            old_states, new_states, new_signs = act_func(
                h, states, data_cache, cached_until, cache_until, ones
            )
            cached_until = cache_until

            n_vals = new_states.shape[0]
            memory_slice = slice(data_idx, data_idx + n_vals)
            types_dict[type_str] = memory_slice
            rows[memory_slice] = old_states
            cols[memory_slice] = new_states
            signs[memory_slice] = new_signs
            if numeric_weights:
                values[memory_slice] = weight * new_signs
            data_idx += n_vals

        return rows, cols, values, signs, types_dict

    def _act_oper_list_cached(
        self, oper_list, states, data_cache, cached_until, cache_until, ones
    ):
        old_states, new_states, signs = data_cache.get("data")

        applied_len = cached_until
        for oper in oper_list[cached_until:]:
            states_alive_filt, new_states, new_parity = self.act_oper(oper, new_states)
            signs = signs[states_alive_filt] * new_parity
            old_states = old_states[states_alive_filt]
            applied_len += 1
            if applied_len == cache_until:
                data_cache["data"] = (old_states, new_states, signs)

        if cache_until == 0:
            data_cache["data"] = (states, states, ones)

        return old_states, new_states, signs

    def _construct_data_repr_restricted(self, fock_basis):
        rows_list, cols_list, pars_list, types_dict, data_idx = [], [], [], {}, 0

        type_strings = [self.oper_list_to_str(h) for h in self.oper_list]

        oper_list = self.oper_list
        weights = self.weights
        states = fock_basis.states

        for h, type_str in zip(oper_list, type_strings):
            old_states, new_states, parities = self.act_oper_list(h, states)
            row_indices, _ = fock_basis.get_indices(old_states)
            col_indices, in_basis = fock_basis.get_indices(new_states)

            rows_list.append(row_indices[in_basis])
            cols_list.append(col_indices)
            pars_list.append(parities[in_basis])

            n_vals = np.count_nonzero(in_basis)
            types_dict[type_str] = slice(data_idx, data_idx + n_vals)
            data_idx += n_vals

        return self._finalize_data_repr(
            rows_list, cols_list, pars_list, types_dict, type_strings, weights, data_idx
        )

    def _finalize_data_repr(
        self,
        rows_list,
        cols_list,
        pars_list,
        types_dict,
        type_strings,
        weights,
        data_idx,
    ):
        if data_idx == 0:
            rows = np.array([], dtype=np.int32)
            cols = np.array([], dtype=np.int32)
            parities = np.array([], dtype=np.int8)
        else:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            parities = np.concatenate(pars_list)

        ### Load values into right positions if weights are not symbolic
        vals = np.zeros(data_idx, dtype=np.complex128)
        numeric_weights = all(not isinstance(w, sympy.Basic) for w in weights)
        if numeric_weights:
            for value, type_str in zip(weights, type_strings):
                data_slice = types_dict[type_str]
                vals[data_slice] = value * parities[data_slice]

        return rows, cols, vals, parities, types_dict
