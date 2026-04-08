import numpy as np
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

from typing import List, Any
from copy import deepcopy

from .base import FermionSequenceBase
from .Mixins.representation import RepresentFermionsMixin
from .Mixins.algebra import FermionAlgebraMixin
from .Mixins.compiler import FermionDataCompilerMixin


class FermionSequence(
    FermionSequenceBase,
    FermionAlgebraMixin,
    FermionDataCompilerMixin,
    RepresentFermionsMixin,
):
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
            oper_list, parsed_weights = self._parse_sympy(operators[0], sympy_operators)
            self.oper_list = oper_list
            self.weights = parsed_weights
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

    @staticmethod
    def sort_cache_friendly(oper_list, weights):
        """
        Sort subsequences in a prefix-aware order that is friendlier to the
        single-prefix compile cache than plain lexicographic sorting.

        The ordering keeps shallow siblings together before descending into
        deeper descendants, e.g. `[5, 4], [5, 6], [5, 4, 1, 0]`.
        """
        root = {"terms": [], "children": {}}

        for seq, weight in zip(deepcopy(oper_list), deepcopy(weights)):
            node = root
            for oper in seq:
                node = node["children"].setdefault(oper, {"terms": [], "children": {}})
            node["terms"].append((seq, weight))

        ordered = []

        def subtree_height(node):
            if len(node["children"]) == 0:
                return 0
            return 1 + min(subtree_height(child) for child in node["children"].values())

        def collapse(node):
            ordered.extend(node["terms"])
            child_items = sorted(
                node["children"].items(),
                key=lambda item: (subtree_height(item[1]), item[0]),
            )

            for _, child in child_items:
                ordered.extend(child["terms"])

            for _, child in child_items:
                child_terms = child["terms"]
                child["terms"] = []
                collapse(child)
                child["terms"] = child_terms

        collapse(root)

        new_oper_list = [seq for seq, _ in ordered]
        new_weights = [weight for _, weight in ordered]
        return new_oper_list, new_weights

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

    ###########################################
    #####      SETTING SUBSEQUENCES       #####
    ###########################################
    _setitem_handlers = {
        slice: "_set_slice",
        tuple: "_set_tuple",
        np.ndarray: "_set_ndarray",
        int: "_set_int",
        str: "_set_str",
    }

    def __setitem__(self, indexing, value: [int, float, complex]):
        if isinstance(indexing, self.__class__):
            old_val, str_list = self._set_opersequence(indexing, value)
        else:
            handler_func_name = self._setitem_handlers.get(
                type(indexing), "_handle_set_default"
            )
            handler_func = getattr(self, handler_func_name)
            old_val, str_list = handler_func(indexing, value)

    def _handle_set_default(self, indexing, value):
        raise TypeError(f"Setting value by {type(indexing).__name__} not supported")

    def _set_slice(self, indexing, value):
        print("Setting value by slice not yet supported")

    def _set_tuple(self, indexing, value):
        print("Setting value by tuple not yet supported")

    def _set_ndarray(self, indexing, value):
        print("Setting value by numpy array not yet supported")

    def _set_int(self, indexing, value):
        print("Setting value by int not yet supported")

    def _set_str(self, indexing, value):
        print("Setting value by str not yet supported")

    def _set_opersequence(self, indexing, value):
        old_value = self.weights[self.oper_list.index(indexing.oper_list[0])]
        str_list = []
        for i in range(len(indexing.oper_list)):
            check_item = indexing.oper_list[i]
            if check_item in self.oper_list:
                self.weights[self.oper_list.index(check_item)] = value
                str_list.append(self.oper_list_to_key(check_item))
        return old_value, str_list

    def __gt__(self, values: [int, float, complex, List, np.ndarray]):
        if isinstance(values, (int, float, complex)):
            self.weights = [values for i in range(len(self.oper_list))]
        else:
            if len(self.oper_list) != len(values):
                raise ValueError(
                    f"Length of input array ({len(values)}) does not match the number of subsequences ({len(self.oper_list)})"
                )
            self.weights = values
        return self

    set_values = __gt__  ## readable alias
