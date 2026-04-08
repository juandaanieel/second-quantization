from ..fock_basis import FockBasis
from ..fermion_data import FermionData

import numpy as np
import sympy


# @TODO - fix clunky compiler structure for cached/uncached options
# @TODO - check efficiency of restricted path mapping of states to indexes
class FermionDataCompilerMixin:
    """
    Mixin implementing compilation of fermionic operator sequences into data
    representations on a chosen Fock basis.

    Expects the host class to provide:
    - `self.oper_list`
    - `self.weights`
    - `self.oper_list_to_key(...)`
    - `self.act_oper(...)`
    - `self.act_oper_list(...)`
    - `self.sort_cache_friendly(...)`
    - `self.is_normal_ordered(...)`
    """

    ####################################################
    #####      CONSTRUCT DATA REPRESENTATION       #####
    ####################################################
    _compile_formats_list = ["coo", "csr", "array"]

    def compile(
        self,
        fock_basis: [int, FockBasis] = None,
        data_format: str = "coo",
        cached: bool = False,
    ):
        """
        Core compiler function of FermionSequence: building the sparse representation of a fermionic Hamiltonian
        in a given fock_basis, wrapped as a FermionData object.
        Compile path distuinghuishes two options:
        - Ordered fock basis vs restricted fock basis
            (in an ordered fock basis the integers representing each state are equal to their position in the matrix, saving an indexing step)
        - Cached construction vs non-cached construction
            (cached mode will store intermediate results for similar operator types, saving time at some temporary memory cost)

        Args:
            fock_basis (Optional):
                if int - interprets as the number of fermionic sites in the fock basis and generates the FockBasis object.
                if FockBasis - uses the existing user-created FockBasis.
                if None - infers the number of unique fermions that appear in the Hamiltonian and constructs FockBasis based on that
            data_format (str): Accepts 'coo','csr','array'. FermionData object is optimized for chosen data format.
            cached (bool): if True, compiler will use cached functionality to construct data representations.

        Returns:
            sparse-data (FermionData): fermion_data
        """
        #######################
        ### Validate inputs ###
        #######################
        if data_format not in self._compile_formats_list:
            raise ValueError(
                f"Selected invalid data_format. "
                f"Implemented data formats are: {self._compile_formats_list}"
            )

        ##########################
        ### Validate FockBasis ###
        ##########################
        ## If given an integer, generate a FockBasis of given size N object for later convenience
        if isinstance(fock_basis, int):
            fock_basis = FockBasis(fock_basis)

        ## If no specific FockBasis was given, infer the size of the fockspace from the Hamiltonian itself
        elif fock_basis is None:
            non_empty_subsequences = [
                sublist for sublist in self.oper_list if len(sublist)
            ]
            inferred_N = (
                (max(max(sublist) for sublist in non_empty_subsequences) >> 1) + 1
                if len(non_empty_subsequences)
                else 0
            )
            fock_basis = FockBasis(inferred_N)

        #############################
        ### Construct FermionData ###
        #############################
        ## Construct the sparse representation object in specified basis
        sparse_data = FermionData(
            fock_basis,
            self._construct_data_repr(fock_basis, cached=cached),
            data_backend=data_format,
        )

        return sparse_data

    def _construct_data_repr(self, fock_basis: FockBasis, cached: bool):
        if fock_basis.is_ordered:
            return self._construct_data_repr_ordered(fock_basis, cached)
        return self._construct_data_repr_restricted(fock_basis, cached)

    def _construct_data_repr_ordered(self, fock_basis: FockBasis, cached: bool):
        """
        Applies the stored hamiltonian to a set of ordered fock states, wrapped in FockBasis object.

        Returns:
            rows (ndarray[int]): row indices of non-zero terms
            cols (ndarray[int]): column indices of non-zero terms
            values (ndarray[complex]): values of each row/col pair
            signs (ndarray[int]): relative signs of operators, required for quick updating of arrays at later stage
            type_dict (dict): a dict pointing to where in the row/col memory the values related to a particular operator are stored
        """
        #######################
        ### Initialize data ###
        #######################
        ## Pre-calculate the amount of data that will be generated
        dim_basis = fock_basis.N
        n_required_data = 0
        for oper in self.oper_list:
            if len(oper):
                k_sites = np.unique(np.array(oper) >> 1).size
            else:
                k_sites = 0
            n_required_data += 2 ** (dim_basis - k_sites)

        ## Create empty arrays with correct size and type
        rows = np.empty(n_required_data, dtype=fock_basis.states.dtype)
        cols = np.empty(n_required_data, dtype=fock_basis.states.dtype)
        signs = np.empty(n_required_data, dtype=np.int8)
        values = np.zeros(
            n_required_data, dtype=np.complex128
        )  ## Should default be complex?
        types_dict, data_idx = {}, 0

        ##################################################
        ###  Unpack objects and prepare sympy metadata ###
        ##################################################
        oper_list, weights = (
            (self.oper_list, self.weights)
            if not cached
            else self.__class__.sort_cache_friendly(self.oper_list, self.weights)
        )
        type_keys = [self.oper_list_to_key(h) for h in oper_list]
        sympy_metadata = self._build_param_metadata(weights)
        update_symbols = set(sympy_metadata["symbols"])
        symbolic_terms = self._prepare_symbolic_terms(weights, update_symbols)

        act_func = self.act_oper_list if not cached else self._act_oper_list_cached
        states = fock_basis.states

        #######################
        ###  Uncached path  ###
        #######################
        if not cached:
            for h, type_key, weight, symbolic_term in zip(
                oper_list, type_keys, weights, symbolic_terms
            ):
                old_states, new_states, new_signs = act_func(
                    h,
                    states,
                )
                n_vals = new_states.shape[0]
                memory_slice = slice(data_idx, data_idx + n_vals)
                types_dict[type_key] = memory_slice
                rows[memory_slice] = old_states
                cols[memory_slice] = new_states
                signs[memory_slice] = new_signs
                if not self._is_updateable_symbolic_weight(weight):
                    values[memory_slice] = weight * new_signs
                else:
                    self._register_symbolic_term(
                        sympy_metadata, symbolic_term, memory_slice
                    )
                data_idx += n_vals
            return rows, cols, values, signs, types_dict, sympy_metadata

        #######################
        ###   Cached path   ###
        #######################
        else:
            ones = np.ones(len(states), dtype=np.int8)
            data_cache = (states, states, ones)
            cache_until_list = []
            for idx in range(len(oper_list) - 1):
                cache_until_list.append(
                    self.__class__.common_prefix_length(
                        oper_list[idx], oper_list[idx + 1]
                    )
                )
            cache_until_list.append(0)
            cached_until = 0

            ## Loop over operators and type string, apply each operator list to set of states
            for h, type_key, weight, cache_until, symbolic_term in zip(
                oper_list, type_keys, weights, cache_until_list, symbolic_terms
            ):
                old_states, new_states, new_signs, data_cache = act_func(
                    h, states, data_cache, cached_until, cache_until, ones
                )
                cached_until = cache_until

                n_vals = new_states.shape[0]
                memory_slice = slice(data_idx, data_idx + n_vals)
                types_dict[type_key] = memory_slice
                rows[memory_slice] = old_states
                cols[memory_slice] = new_states
                signs[memory_slice] = new_signs
                if not self._is_updateable_symbolic_weight(weight):
                    values[memory_slice] = weight * new_signs
                else:
                    self._register_symbolic_term(
                        sympy_metadata, symbolic_term, memory_slice
                    )
                data_idx += n_vals

            return rows, cols, values, signs, types_dict, sympy_metadata

    def _act_oper_list_cached(
        self, oper_list, states, data_cache, cached_until, cache_until, ones
    ):
        old_states, new_states, signs = data_cache
        should_update_cache = cache_until > cached_until

        applied_len = cached_until
        for oper in oper_list[cached_until:]:
            states_alive_filt, new_states, new_parity = self.act_oper(oper, new_states)
            signs = signs[states_alive_filt] * new_parity
            old_states = old_states[states_alive_filt]
            applied_len += 1
            if should_update_cache and applied_len == cache_until:
                data_cache = (old_states, new_states, signs)

        if cache_until == 0:
            data_cache = (states, states, ones)

        return old_states, new_states, signs, data_cache

    def _construct_data_repr_restricted(self, fock_basis: FockBasis, cached: bool):
        """
        Applies the stored hamiltonian to a restricted set of fock states, wrapped
        in a FockBasis object.

        Returns:
            rows (ndarray[int]): row indices of non-zero terms
            cols (ndarray[int]): column indices of non-zero terms
            values (ndarray[complex]): values of each row/col pair
            signs (ndarray[int]): relative signs of operators, required for quick
                updating of arrays at later stage
            type_dict (dict): a dict pointing to where in the row/col memory the
                values related to a particular operator are stored
        """
        #######################
        ### Initialize data ###
        #######################
        rows_list, cols_list, pars_list, types_dict, data_idx = [], [], [], {}, 0

        #######################
        ###  Unpack objects ###
        #######################
        oper_list, weights = (
            (self.oper_list, self.weights)
            if not cached
            else self.__class__.sort_lexicographical(self.oper_list, self.weights)
        )
        type_keys = [self.oper_list_to_key(h) for h in oper_list]
        sympy_metadata = self._build_param_metadata(weights)
        update_symbols = set(sympy_metadata["symbols"])
        symbolic_terms = self._prepare_symbolic_terms(weights, update_symbols)
        act_func = self.act_oper_list if not cached else self._act_oper_list_cached
        states = fock_basis.states

        #######################
        ###  Uncached path  ###
        #######################
        if not cached:
            for h, type_key, weight, symbolic_term in zip(
                oper_list, type_keys, weights, symbolic_terms
            ):
                old_states, new_states, parities = act_func(h, states)
                row_indices, _ = fock_basis.get_indices(old_states)
                col_indices, in_basis = fock_basis.get_indices(new_states)

                rows_list.append(row_indices[in_basis])
                cols_list.append(col_indices)
                pars_list.append(parities[in_basis])

                n_vals = np.count_nonzero(in_basis)
                memory_slice = slice(data_idx, data_idx + n_vals)
                types_dict[type_key] = memory_slice
                if self._is_updateable_symbolic_weight(weight) and n_vals > 0:
                    self._register_symbolic_term(
                        sympy_metadata, symbolic_term, memory_slice
                    )
                data_idx += n_vals

        #######################
        ###   Cached path   ###
        #######################
        else:
            ones = np.ones(len(states), dtype=np.int8)
            data_cache = (states, states, ones)
            cache_until_list = []
            for idx in range(len(oper_list) - 1):
                cache_until_list.append(
                    self.__class__.common_prefix_length(
                        oper_list[idx], oper_list[idx + 1]
                    )
                )
            cache_until_list.append(0)
            cached_until = 0

            for h, type_key, weight, cache_until, symbolic_term in zip(
                oper_list, type_keys, weights, cache_until_list, symbolic_terms
            ):
                old_states, new_states, parities, data_cache = act_func(
                    h, states, data_cache, cached_until, cache_until, ones
                )
                cached_until = cache_until
                row_indices, _ = fock_basis.get_indices(old_states)
                col_indices, in_basis = fock_basis.get_indices(new_states)

                rows_list.append(row_indices[in_basis])
                cols_list.append(col_indices)
                pars_list.append(parities[in_basis])

                n_vals = np.count_nonzero(in_basis)
                memory_slice = slice(data_idx, data_idx + n_vals)
                types_dict[type_key] = memory_slice
                if self._is_updateable_symbolic_weight(weight) and n_vals > 0:
                    self._register_symbolic_term(
                        sympy_metadata, symbolic_term, memory_slice
                    )
                data_idx += n_vals

        return self._finalize_data_repr(
            rows_list,
            cols_list,
            pars_list,
            types_dict,
            type_keys,
            weights,
            data_idx,
            sympy_metadata,
        )

    def _finalize_data_repr(
        self,
        rows_list,
        cols_list,
        pars_list,
        types_dict,
        type_keys,
        weights,
        data_idx,
        sympy_metadata,
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
        for value, type_key in zip(weights, type_keys):
            if self._is_updateable_symbolic_weight(value):
                continue
            data_slice = types_dict[type_key]
            if data_slice.stop > data_slice.start:
                vals[data_slice] = value * parities[data_slice]

        return rows, cols, vals, parities, types_dict, sympy_metadata

    @staticmethod
    def common_prefix_length(a, b):
        """
        Return the length of the common prefix shared by two operator lists.
        """
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    @staticmethod
    def _build_param_metadata(weights):
        update_symbols = set()
        for weight in weights:
            if isinstance(weight, sympy.Basic):
                update_symbols.update(weight.free_symbols)

        return {
            "symbols": tuple(sympy.ordered(update_symbols)),
            "param_dict": {symbol: [] for symbol in sympy.ordered(update_symbols)},
            "symbolic_terms": [],
            "param_values": {symbol: 0.0 for symbol in sympy.ordered(update_symbols)},
        }

    @classmethod
    def _prepare_symbolic_terms(cls, weights, update_symbols):
        symbolic_terms = []
        for weight in weights:
            if cls._is_updateable_symbolic_weight(weight):
                symbolic_terms.append(
                    cls._classify_symbolic_weight(weight, update_symbols)
                )
            else:
                symbolic_terms.append(None)
        return symbolic_terms

    @staticmethod
    def _register_symbolic_term(sympy_metadata, symbolic_term, memory_slice):
        if symbolic_term is None:
            return

        symbolic_term = symbolic_term.copy()
        symbolic_term["slice"] = memory_slice
        term_idx = len(sympy_metadata["symbolic_terms"])
        sympy_metadata["symbolic_terms"].append(symbolic_term)

        for symbol in symbolic_term["symbols"]:
            sympy_metadata["param_dict"][symbol].append(term_idx)

    @staticmethod
    def _classify_symbolic_weight(weight, update_symbols):
        if not isinstance(weight, sympy.Basic):
            return None

        free_symbols = tuple(sympy.ordered(weight.free_symbols))
        if len(free_symbols) == 0:
            return None

        term_data = {"expr": weight, "symbols": free_symbols}

        if len(free_symbols) == 1:
            symbol = free_symbols[0]
            prefactor = sympy.simplify(weight / symbol)
            if prefactor.free_symbols.isdisjoint(update_symbols):
                term_data["fast_symbol"] = symbol
                term_data["fast_factor"] = prefactor

        return term_data

    @staticmethod
    def _is_updateable_symbolic_weight(weight):
        return isinstance(weight, sympy.Basic) and len(weight.free_symbols) > 0
