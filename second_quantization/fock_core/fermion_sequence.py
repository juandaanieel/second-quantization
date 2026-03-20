from copy import deepcopy
from collections import Counter

import numpy as np
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from typing import List, Self, Any

from .base import FermionSequenceBase
from .fock_basis import FockBasis
from .fermion_data import FermionData
from .Mixins.representation import RepresentFermionsMixin
from .Mixins.algebra import FermionAlgebraMixin


def common_prefix_length(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class FermionSequence(FermionSequenceBase, FermionAlgebraMixin, RepresentFermionsMixin):
    """
    Represents sequences of fermionic operators and handles common arithmetic operations.
    Provides an interface to retrieve matrix or sparse representations of the sequence in any general fock basis.
    Args:
        oper_list (list[list[int]]): a nested list storing the integer representations of fermionic operators in a sequence.
        weights (list[complex]): a list of values storing the weight of each subsequence of operators.
        bypass_parse (bool (default: False)): flag to indicate whether the input weights and oper_list should be stored directly if they are already in the correct format (used for internal functions to save time by avoiding parsing).

    Attributes:
        _sympy_operators (optional):
            Internal representation of the operator sequence in SymPy form.
            Only set when the object is initialized via SymPy-based constructors
            or conversion function.
        _sympy_dict (optional):
            Internal dictionary mapping symbolic prefactors to their corresponding operator subsequences.

    Notes:
        The integer representation of operators is as follows:
            - Even integers (0, 2, 4, ...) represent creation operators, where the integer divided by 2 gives the site index.
            - Odd integers (1, 3, 5, ...) represent annihilation operators, where the integer minus 1 divided by 2 gives the site index.
        Normal ordering assumes that creation operators should be ordered before annihilation operators, and within each type, operators are ordered by their site index.
        By default, each site is assumed to have two spin states (up and down). The labelling of each fermion can be updated by the _sympy_operators attribute or .set_ordering() (@todo).
    """

    #####################################
    #####      INITIALIZATION       #####
    #####################################
    def __init__(
        self,
        *operators: [List, tuple, int, sympy.Basic],
        weights: List = None,
        bypass_parse: bool = False,
        sympy_operators: List[FermionOp] = None,
    ):
        """
        Initialise a sequence of operators with specified weights.
        Args:
            *operators ([List[int], int, tuple]): a variable length argument list of operators in the sequence, where each operator can be represented as a list of integers, a string, an integer, or a tuple
            weights (List): a list of values storing the weight of each subsequence of operators. If not provided, all weights are set to 1 by default. Weights can be int, float, complex, or sympy expressions.
            bypass_parse (bool): flag to indicate whether the input weights and oper_list should be stored directly if they are already in the correct format (used for internal functions to save time by avoiding parsing)
        """
        ## For internally created FermionSequence objects, directly load data without parsing as format can be assumed to be correct
        if bypass_parse:
            self.oper_list = operators[0]
            self.weights = weights if weights is not None else [1] * len(self.oper_list)
            return

        ## Empty constructor just sets empty lists
        if len(operators) == 0:
            self.oper_list = []
            self.weights = []
            return

        ## Sympy constructor path
        if len(operators) == 1 and isinstance(operators[0], sympy.Basic):
            oper_list, parsed_weights, sympy_dict = self._parse_sympy(
                operators[0], sympy_operators
            )
            self.oper_list = oper_list
            self.weights = parsed_weights
            self._sympy_dict = sympy_dict
            self._sympy_operators = sympy_operators
            return

        ## Regular constructor path
        parsed_operators = self.parse_operator_input(*operators)
        self.oper_list = parsed_operators

        if weights is not None and len(weights) != len(parsed_operators):
            raise ValueError(
                "Length of provided weights does not match number of subsequences"
            )

        self.weights = weights if weights is not None else [1] * len(self.oper_list)

    ##################################
    #####      Conversions       #####
    ##################################
    def to_sympy(self):
        """
        Converts FermionSequence to a SymPy expression of FermionOps.
        If the FermionSequence was initialized from a SymPy expression, retrieves the original expression and operator list from the internal attributes.
        Otherwise, creates Sympy FermionOp objects based on the FermionSequence default ordering of operators in oper_list and constructs the corresponding SymPy expression.s
        Returns:
            H_sympy (Expr): The SymPy expression representing the fermionic Hamiltonian.
            operators (List[FermionOp]): The list of SymPy fermionic operators.
        """
        ### Generate list of new FermionOps of not yet present
        if not hasattr(self, "_sympy_operators") or self._sympy_operators is None:
            operators = []
            max_n = max(max(sublist) for sublist in self.oper_list if len(sublist)) >> 1
            spins = ["\\downarrow", "\\uparrow"]

            for op in range(0, max_n + 1, 1):
                site = op // 2
                spin = op % 2

                operators.append(FermionOp("c_{" + f"{site},{spins[spin]}" + "}"))
        ### Otherwise, retrieve the original operator list from the internal attribute
        else:
            operators = self._sympy_operators

        ### Pass each operator sequence and weight to construct the full SymPy expression, using the original operator list to ensure correct labelling of operators
        H_sympy = 0
        for weight, opers in zip(self.weights, self.oper_list):
            term = weight
            for op in reversed(opers):
                idx = op // 2
                term *= Dagger(operators[idx]) if op % 2 == 0 else operators[idx]

            H_sympy += term

        return H_sympy, operators

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
            inferred_N = (
                max(max(sublist) for sublist in self.oper_list if len(sublist)) >> 1
            ) + 1
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
        values = np.zeros(n_required_data, dtype=np.float64)

        types_dict, data_idx = {}, 0
        type_strings = [self.oper_list_to_str(h) for h in self.oper_list]

        ## Unpack reference to array and weights (saves time inside the loop for larger systems)
        oper_list, weights = self.__class__.sort_lexicographical(
            self.oper_list, self.weights
        )
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
        if len(weights) > 0 and not isinstance(weights[0], sympy.Basic):
            for value, type_str in zip(weights, type_strings):
                data_slice = types_dict[type_str]
                vals[data_slice] = value * parities[data_slice]

        return rows, cols, vals, parities, types_dict

    ######################################################
    #####             OPERATOR ARITHMETIC            #####
    ######################################################

    ## Specify the non-FermionSequence types that are allowed to be added/multiplied to an FermionSequence
    _allowed_types = (int, float, complex, sympy.Basic)

    def __add__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'"
            )

        new_weights, new_oper_list = [], []
        new_weights.extend(self.weights)
        new_oper_list.extend(self.oper_list)
        if isinstance(oper_sequence, self.__class__):
            new_weights.extend(oper_sequence.weights)
            new_oper_list.extend(oper_sequence.oper_list)
        else:
            new_weights.append(oper_sequence)
            new_oper_list.append([])  ## Empty list corresponds to identity operator

        result = FermionSequence(new_oper_list, weights=new_weights, bypass_parse=True)
        result.merge_terms()
        return result

    def __iadd__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'"
            )

        if isinstance(oper_sequence, self.__class__):
            self.weights.extend(oper_sequence.weights)
            self.oper_list.extend(oper_sequence.oper_list)

        else:
            self.weights.append(oper_sequence)
            self.oper_list.append([])  ## Empty list corresponds to identity operator

        self.merge_terms()
        return self

    def __sub__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'"
            )

        new_weights = []
        new_oper_list = []
        new_weights.extend(self.weights)
        new_oper_list.extend(self.oper_list)
        if isinstance(oper_sequence, self.__class__):
            new_weights.extend([-1 * w for w in oper_sequence.weights])
            new_oper_list.extend(oper_sequence.oper_list)

        else:
            new_weights.append(-oper_sequence)
            new_oper_list.append([])  ## Empty list corresponds to identity operator

        result = FermionSequence(new_oper_list, weights=new_weights, bypass_parse=True)
        result.merge_terms()
        return result

    def __isub__(self, oper_sequence):
        if not isinstance(oper_sequence, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for +: {self.__class__.__name__} and '{type(oper_sequence).__name__}'"
            )

        if isinstance(oper_sequence, self.__class__):
            self.weights.extend([-1 * w for w in oper_sequence.weights])
            self.oper_list.extend(oper_sequence.oper_list)
        else:
            self.weights.append(-oper_sequence)
            self.oper_list.append([])  ## Empty list corresponds to identity operator
        self.merge_terms()
        return self

    def __pos__(self) -> Self:
        return self

    def __neg__(self):
        new_weights = [-1 * w for w in self.weights]
        negative_oper = FermionSequence(
            deepcopy(self.oper_list), weights=new_weights, bypass_parse=True
        )
        return negative_oper

    def __pow__(self, exponent: int):
        if not isinstance(exponent, int):
            return self
        product = deepcopy(self)
        for _ in range(1, exponent):
            product *= self
        return product

    def __ipow__(self, exponent: int) -> Self:
        if not isinstance(exponent, int):
            return self
        for _ in range(1, exponent):
            self *= self
        return self

    def __mul__(
        self, multiplier: ["FermionSequence", int, float, complex]
    ) -> "FermionSequence":
        if not isinstance(multiplier, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for *: {self.__class__.__name__} and '{type(multiplier).__name__}'"
            )

        ### For multiplying by a scalar/Symbol, multiply the weights and return a new FermionSequence
        if not isinstance(multiplier, self.__class__):
            new_weights = [w * multiplier for w in self.weights]
            product = FermionSequence(
                self.oper_list, weights=new_weights, bypass_parse=True
            )
            return product

        ### Otherwise, multiply sequences together and clean the result
        product_outcome = self._multiplication_basis(
            self.weights, self.oper_list, multiplier.weights, multiplier.oper_list
        )
        product = FermionSequence(
            product_outcome[1], weights=product_outcome[0], bypass_parse=True
        )
        product.clean()
        return product

    def __rmul__(self, multiplier) -> "FermionSequence":
        if not isinstance(multiplier, self._allowed_types):
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(multiplier).__name__}' and {self.__class__.__name__}"
            )
        new_weights = [w * multiplier for w in self.weights]
        product = FermionSequence(
            self.oper_list, weights=new_weights, bypass_parse=True
        )
        return product

    def __imul__(self, multiplier) -> Self:
        if not isinstance(multiplier, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for *: {self.__class__.__name__} and '{type(multiplier).__name__}'"
            )

        ### For multiplying by a scalar/Symbol, multiply the weights and return a new FermionSequence
        if not isinstance(multiplier, self.__class__):
            self.weights = [w * multiplier for w in self.weights]
            return self
        ### Otherwise, multiply sequences together and clean the result
        product_outcome = self._multiplication_basis(
            self.weights, self.oper_list, multiplier.weights, multiplier.oper_list
        )
        self.oper_list = product_outcome[1]
        self.weights = product_outcome[0]
        self.clean()
        return self

    def commute(self, operator_2):
        commutator = self * operator_2 - operator_2 * self
        commutator.remove_zero_weight()
        return commutator

    def anticommute(self, operator_2):
        anticommutator = self * operator_2 + operator_2 * self
        anticommutator.remove_zero_weight()
        return anticommutator

    def __truediv__(self, divisor: [int, float]) -> "FermionSequence":
        """
        / operator
        Divides the weights of the FermionSequence by divisor, returns new FermionSequence
        """
        if not isinstance(divisor, self._allowed_types):
            raise TypeError(
                f"Unsupported operand type(s) for /: {self.__class__.__name__} and '{type(divisor).__name__}'"
            )
        oper_list = self.oper_list[:]
        new_weights = [w / divisor for w in self.weights]
        return FermionSequence(oper_list, weights=new_weights, bypass_parse=True)

    def __itruediv__(self, divisor: [int, float]) -> Self:
        """
        in-place / operator
        Divides the weights of the FermionSequence by divisor
        """
        if not isinstance(divisor, self._allowed_types):
            raise TypeError(
                f"Unsupported operand type(s) for /: {self.__class__.__name__} and '{type(divisor).__name__}'"
            )
        self.weights = [w / divisor for w in self.weights]
        return self

    @staticmethod
    def _multiplication_basis(weights_1, oper_list_1, weights_2, oper_list_2):
        """
        Basis for multiplying two FermionSequences.
        Loops through the nested operator lists and concatenates their sublists.
        Cleaning of the expression (i.e. normal ordering) is required after this process.
        """
        oper_products = []
        product_weights = []
        for idx_1, oper_seq_1 in enumerate(oper_list_1):
            for idx_2, oper_seq_2 in enumerate(oper_list_2):
                temp = oper_seq_2.copy()
                temp.extend(oper_seq_1)
                oper_products.append(temp)
                product_weights.append(weights_1[idx_1] * weights_2[idx_2])

        return product_weights, oper_products

    def __mod__(self, modulo: int):
        """
        % operator
        Returns True if the length of each subsequence satisfies length % modulo = 0,
        returns False otherwise.
        """
        if not isinstance(modulo, int):
            return False
        for oper_seq in self.oper_list:
            if isinstance(oper_seq, list):
                if len(oper_seq) % modulo != 0:
                    return False
        return True

    def __invert__(self) -> "FermionSequence":
        """
        ~ operator
        Returns a new FermionSequence representing the complex conjugate of the original sequence
        Note: Does not normal order the new sequence before returning.
        """
        invert_list = []
        flip_bit = 0b1
        for oper_seq in deepcopy(self.oper_list):
            new_opers = []
            if not isinstance(oper_seq, list):
                if isinstance(oper_seq, complex):
                    invert_list.append(oper_seq.conjugate())
                    continue
                invert_list.append(oper_seq)
                continue

            for oper in oper_seq:
                conjugate_oper = oper ^ flip_bit
                new_opers.append(conjugate_oper)
            new_opers.reverse()
            invert_list.append(new_opers)

        inverse = FermionSequence(
            invert_list,
            weights=[
                w.conjugate() if isinstance(w, (complex, sympy.Basic)) else w
                for w in self.weights
            ],
            bypass_parse=True,
        )
        return inverse

    dag = __invert__  ## Alias for complex conjugation

    def __rshift__(self, shift_by: int) -> "FermionSequence":
        """
        >> operator.
        Increments the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float, np.ndarray, list)):
            raise TypeError(
                f"Unsupported operand type(s) for >>: {self.__class__.__name__} and '{type(shift_by).__name__}'"
            )

        if isinstance(shift_by, (int, float)):
            new_opers = []
            for oper_seq in deepcopy(self.oper_list):
                for i in range(len(oper_seq)):
                    oper_seq[i] += (int(shift_by) * 2) << 1
                new_opers.append(oper_seq)
            shifted_sequence = FermionSequence(
                new_opers, weights=self.weights[:], bypass_parse=True
            )
            return shifted_sequence
        else:
            base_sequence = FermionSequence(
                deepcopy(self.oper_list), weights=self.weights[:], bypass_parse=True
            )
            shift_sequence = FermionSequence(
                deepcopy(self.oper_list), weights=self.weights[:], bypass_parse=True
            )
            for shift in shift_by:
                shift_sequence += base_sequence >> int(shift)
            return shift_sequence

    extend = __rshift__

    def __irshift__(self, shift_by) -> Self:
        """
        In-place >> operator.
        Increments the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float)):
            raise TypeError(
                f"Unsupported operand type(s) for >>=: {self.__class__.__name__} and '{type(shift_by).__name__}'"
            )

        for oper_seq in self.oper_list:
            for i in range(len(oper_seq)):
                oper_seq[i] += (int(shift_by) * 2) << 1
        return self

    def __lshift__(self, shift_by: int) -> "FermionSequence":
        """
        << operator.
        Decrements the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float)):
            raise TypeError(
                f"Unsupported operand type(s) for <<: {self.__class__.__name__} and '{type(shift_by).__name__}'"
            )

        new_opers = []
        for oper_seq in deepcopy(self.oper_list):
            for i in range(len(oper_seq)):
                oper_seq[i] -= (int(shift_by) * 2) << 1
            new_opers.append(oper_seq)
        shifted_sequence = FermionSequence(
            new_opers, weights=self.weights[:], bypass_parse=True
        )
        return shifted_sequence

    def __ilshift__(self, shift_by: int) -> Self:
        """
        In-place << operator.
        Decrements the site that each operator in oper_list acts on.

        Args:
            shift_by (int): the amount of sites to decrease by
        """
        if not isinstance(shift_by, (int, float)):
            raise TypeError(
                f"Unsupported operand type(s) for <<=: {self.__class__.__name__} and '{type(shift_by).__name__}'"
            )

        for oper_seq in self.oper_list:
            for i in range(len(oper_seq)):
                oper_seq[i] -= (int(shift_by) * 2) << 1
        return self

    ###########################################################
    #####     ORDERING AND CLEANING OPERATOR SEQUENCE     #####
    ###########################################################
    def simplify(self) -> None:
        """
        Calls .expand() then .simplify() on all Sympy expressions in self.weights.

            Returns: None (FermionSequence is modified in-place)
        """
        for idx, weight in enumerate(self.weights):
            if isinstance(weight, sympy.Basic):
                self.weights[idx] = sympy.simplify(sympy.expand(weight))
        self.remove_zero_weight()

    def remove_zero_weight(self) -> None:
        """
        Removes any terms in oper_list that have zero weight.
        Numerical terms are removed rounded to 13 digits.
        Symbolic terms rely on Sympy's == 0 check.

            Returns: None (FermionSequence is modified in-place)
        """
        for idx in range(len(self.weights) - 1, -1, -1):
            if isinstance(self.weights[idx], (int, float, complex)):
                if abs(np.round(self.weights[idx], 13)) == 0.0:
                    self.weights.pop(idx)
                    self.oper_list.pop(idx)
            elif isinstance(self.weights[idx], (sympy.Basic)):
                if self.weights[idx] == 0:
                    self.weights.pop(idx)
                    self.oper_list.pop(idx)

    def remove_duplicates(self) -> None:
        """
        Removes any operator subsequence in oper_list with duplicate operators.
        This enforces Pauli exclusion -> c_i^2 = 0.

            Returns: None (FermionSequence is modified in-place)
        """
        for idx in range(len(self.weights) - 1, -1, -1):
            if len(self.oper_list[idx]) != len(set(self.oper_list[idx])):
                self.oper_list.pop(idx)
                self.weights.pop(idx)

    def merge_terms(self) -> None:
        """
        Merges any identical subsequences, adding their weights.

            Returns: None (FermionSequence is modified in-place)
        """
        merged_weight, merged_list = [], []

        for seq, weight in zip(self.oper_list, self.weights):
            if seq not in merged_list:
                merged_list.append(seq)
                merged_weight.append(weight)
            else:
                get_idx = merged_list.index(seq)
                merged_weight[get_idx] += weight
        self.weights, self.oper_list = merged_weight, merged_list

    def normal_order(self):
        """
        Places the operators in normal order while accounting for minus signs due to swapping of operators
        and removal of duplicate creation/annihilation operators.

            Returns: None (FermionSequence is modified in-place)
        """
        is_normal_ordered = False
        while not is_normal_ordered:
            is_normal_ordered = True
            for seq_idx, oper_seq in enumerate(self.oper_list):
                n = len(oper_seq)
                for i in range(n - 1):
                    flag_swap = False
                    for j in range(n - 1 - i):
                        ## Check if the two operators are in the correct order, if not swap them and update the weight accordingly
                        if (oper_seq[j] % 2) > (oper_seq[j + 1] % 2):
                            continue

                        if oper_seq[j] < oper_seq[j + 1] or (oper_seq[j] % 2) < (
                            oper_seq[j + 1] % 2
                        ):
                            is_normal_ordered = False
                            ## If the two operators are the same, remove them
                            if (oper_seq[j] ^ 0b1) == oper_seq[j + 1]:
                                self.oper_list.append(oper_seq[:])
                                self.weights.append(self.weights[seq_idx] * -1)
                                self.oper_list[seq_idx] = (
                                    oper_seq[:j] + oper_seq[j + 2 :]
                                )
                                self.oper_list[-1][j], self.oper_list[-1][j + 1] = (
                                    self.oper_list[-1][j + 1],
                                    self.oper_list[-1][j],
                                )
                                flag_swap = False
                                ## If the swap resulted in an empty list, remove the term entirely
                                if len(self.oper_list[seq_idx]) == 0:
                                    self.weights.append(self.weights[seq_idx])
                                    self.weights.pop(seq_idx)
                                    self.oper_list.pop(seq_idx)
                                    self.oper_list.append([])
                                break
                            ## If the two operators are not the same, swap them and update the weight by a minus sign
                            else:
                                flag_swap = True
                                oper_seq[j], oper_seq[j + 1] = (
                                    oper_seq[j + 1],
                                    oper_seq[j],
                                )
                                self.weights[seq_idx] *= -1

                    ## If no swap took place, list is sorted
                    if not flag_swap:
                        break

    def clean(self):
        """
        Shortcut for cleaning an operator sequence in the correct order.
        Not following this order will lead to wrong results (e.g. by calling remove_duplicates() before normal_order()).
        @TODO: make an external entry point for the below functions that does some checks so that a user can not make this mistake?
                e.g. by calling is_normal_ordered() before calling remove_duplicates(), while skipping those checks internally.

            Returns: None (FermionSequence is modified in-place)
        """
        self.normal_order()
        self.merge_terms()
        self.remove_duplicates()
        self.remove_zero_weight()

    def sort(self) -> None:
        """
        Sort subsequences lexicographically by their internal operator-list order.

            Returns: None (FermionSequence is modified in-place)
        """
        self.oper_list, self.weights = self.sort_lexicographical(
            self.oper_list, self.weights
        )

    @staticmethod
    def sort_lexicographical(oper_list, weights) -> None:
        """
        Sort subsequences lexicographically by their internal operator-list order.
        This keeps terms with common prefixes adjacent, which is useful for compile-time
        prefix caching.

            Returns: None (FermionSequence is modified in-place)
        """
        combined = list(zip(deepcopy(oper_list), deepcopy(weights)))
        combined.sort(key=lambda x: tuple(x[0]))
        oper_list = [seq for seq, _ in combined]
        weights = [weight for _, weight in combined]
        return oper_list, weights

    #####################################
    #####    Boolean Properties     #####
    #####################################
    def is_normal_ordered(self):
        """
        Checks if the operator sequence is in normal order
        Returns True if the sequence is in normal order, False otherwise
        """
        for oper_seq in self.oper_list:
            for i in range(len(oper_seq) - 1):
                if (oper_seq[i] % 2) < (oper_seq[i + 1] % 2):
                    return False
                elif (oper_seq[i] < oper_seq[i + 1]) and (
                    (oper_seq[i] % 2) < (oper_seq[i + 1] % 2)
                ):
                    return False
        return True

    def is_hermitian(self):
        """
        Checks if the operator sequence is Hermitian.
        Returns True if the sequence is Hermitian, False otherwise.
        """
        conj = ~self
        conj.normal_order()
        return conj == self

    ##############################################
    ######       COMPARISON STATEMENTS       #####
    ##############################################
    def _normalized(self) -> List[tuple]:
        """
        Creates a tuple format of the oper_list and weight attributes, that allows
        for checking equality/inequality in __eq__ and __ne___.
        """
        normalized_operator_list = [tuple(sub) for sub in self.oper_list]
        normalized_sequence = [
            (w, o) for w, o in zip(self.weights, normalized_operator_list)
        ]
        return normalized_sequence

    def __eq__(self, oper_sequence: "FermionSequence") -> bool:
        """
        Check equality of two FermionSequences.
        Two sequences are determined equal if they have identical subsequences with the same weights,
        regardless of the order they appear in.
        """
        if not isinstance(oper_sequence, FermionSequence):
            return False
        return Counter(self._normalized()) == Counter(oper_sequence._normalized())

    def __ne__(self, oper_sequence: "FermionSequence") -> bool:
        """
        Check inequality of two FermionSequence.
        Two sequences are determined non-equal if they do not have identical subsequences,
        regardless of the order they appear in.
        """
        if not isinstance(oper_sequence, "FermionSequence"):
            return True
        return Counter(self._normalized()) != Counter(oper_sequence.normalized())

    ###########################################
    #####      GETTING SUBSEQUENCES       #####
    ###########################################
    _getitem_handlers = {
        slice: "_get_slice",
        tuple: "_get_tuple",
        int: "_get_int",
    }

    def __contains__(self, subsequence):
        return all(item in self.oper_list for item in subsequence.oper_list)

    def __getitem__(self, indexing) -> "FermionSequence":
        handler_func_name = self._getitem_handlers.get(
            type(indexing), "_handle_get_default"
        )
        handler_func = getattr(self, handler_func_name)
        return handler_func(indexing)

    def _handle_get_default(self, indexing: Any):
        raise TypeError(
            f"Getting subsequence by {type(indexing).__name__} not supported"
        )

    def _get_slice(self, indexing: slice):
        subsequence = FermionSequence(
            deepcopy(self.oper_list)[indexing],
            weights=deepcopy(self.weights)[indexing],
            bypass_parse=True,
        )
        return subsequence

    def _get_tuple(self, indexing: tuple):
        new_list = deepcopy(self.oper_list)
        subsequence = FermionSequence(
            [new_list[i] for i in indexing],
            weights=[self.weights[i] for i in indexing],
            bypass_parse=True,
        )
        return subsequence

    def _get_int(self, indexing: int):
        subsequence = FermionSequence(
            [deepcopy(self.oper_list[indexing])],
            weights=[self.weights[indexing]],
            bypass_parse=True,
        )
        return subsequence
