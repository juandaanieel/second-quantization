from typing import List
import sympy
import numpy as np


def operator_verbose(type_str: str, position: int, spin: str):
    """
    Verbose method for constructing an operator, passes the call on to
    self.construct_operator
    Args:
        type (str): specify whether to make an annihilation or creation operator
        position (int): the fermionic site to act on
        spin (str): 'up' or 'down'
    Returns:
        oper (int): integer representing the operator
    """
    annihilation = type_str.startswith("a")
    spin = 0 if (spin.startswith("d")) else 1
    oper = construct_operator(position, spin, annihilation)
    return oper


def operator_from_string(type_str):
    """
    Method for constructing an operator from a string sequence, implementing
    the convention utilized by OpenFermion. Passes a
    correctly formatted call on to construct_operator()
    """
    position = int(type_str[0]) // 2
    spin = int(type_str[0]) & 0b1
    creation = 0 if type_str[-1] == "^" else 1
    return construct_operator(position, spin, creation)


def construct_operator(position: int, spin: int, annihilation: bool = True):
    """
    Creates the integer representing an operator.
    The smallest bit is set or unset depending on whether it stores a creation or annihilation operator
    The remaining bits store the bit position of the operator to act on, which depends on position and spin
    Args:
        position (int): fermionic site of operator to act on
        spin (int): 0 = spin down, 1 = spin up
        annihilation(bool): True = annihilation, False = creation

    Returns:
        oper (int): integer representing an operator
    """
    shift = ((position << 1) + spin) << 1
    oper = shift + annihilation
    return oper


class FermionSequenceBase:
    ##################
    ## Parse Inputs ##
    ##################
    _parseinput_handlers = {
        int: "_parse_int",
        str: "_parse_str",
        list: "_parse_list",
        tuple: "_parse_tuple",
        sympy.Basic: "_parse_sympy",
    }

    def parse_operator_input(
        self, *operators: tuple[List[int], str, int, tuple]
    ) -> List[List[int]]:
        output_list = []
        for op in operators:
            handler_func = self._handle_parse_default
            for key in self._parseinput_handlers.keys():
                if isinstance(op, key):
                    handler_func_name = self._parseinput_handlers.get(
                        key,
                    )
                    break
            handler_func = getattr(self, handler_func_name)
            output_list.append(handler_func(op))
        return output_list

    @staticmethod
    def _parse_sympy(operators_sympy: sympy.Basic) -> List[List]:
        print("Not yet implemented")
        return

    @staticmethod
    def _handle_parse_default(operators):
        raise TypeError(
            f"Invalid input for operators: expected a sequence of Fermion operators, ints or tuples. "
            f"Got {type(operators).__name__} instead."
        )

    @staticmethod
    def _parse_int(operator_int: int) -> List[List]:
        return []

    @staticmethod
    def _parse_str(operator_str: str) -> List[int]:
        split_str = operator_str.split(",")
        parsed_str = [operator_from_string(s) for s in split_str]
        parsed_str.reverse()
        return parsed_str

    @staticmethod
    def _parse_list(operator_list: List[int]) -> List[int]:
        for item in operator_list:
            if isinstance(item, list):
                raise TypeError("Nested lists can not be parsed as input type")
        output = list(operator_list)
        output.reverse()
        return output

    @staticmethod
    def _parse_tuple(operator_tuple: tuple[int, str]) -> List[int]:
        if isinstance(operator_tuple[0], tuple):
            output = []
            for subop_tuple in operator_tuple:
                for item in subop_tuple:
                    if isinstance(item, tuple):
                        raise TypeError(
                            "Not possible to parse a nested tuple with dimension > 2"
                        )
                output.append(operator_verbose(*subop_tuple))
            output.reverse()
            return output
        else:
            return [operator_verbose(*operator_tuple)]

    @staticmethod
    def oper_list_to_str(oper_list):
        """
        Convert a list of integers representing operators
        to a string representation
        """
        return ".".join(map(str, oper_list))

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

    def __call__(self, value_dict):
        if not hasattr(self, "_sympy_dict"):
            raise AttributeError(
                "The FermionSequence object does not have the necessary attributes to set values by calling. Make sure the object was initialized from a sympy expression or that the _sympy_dict attribute is properly set."
            )
        if not hasattr(self, "basis_dict"):
            raise AttributeError(
                "No sparse representation has been initiated for this FermionSequence object. Make sure to initialize a sparse representation for the relevant basis before setting values by calling."
            )
        for sympy_key, new_value in value_dict.items():
            if sympy_key not in self._sympy_dict.keys():
                raise KeyError(
                    f"SymPy key {sympy_key} not found in the FermionSequence's internal dictionary. Make sure to only set values for keys that are already present in the dictionary."
                )
            old_value = self._sympy_dict[sympy_key][0]
            sympy_to_opers_str_list = self._sympy_dict[sympy_key][1]
            self._sympy_dict[sympy_key] = (new_value, sympy_to_opers_str_list)
            if hasattr(self, "basis_dict"):
                for _, sparse_representation in self.basis_dict.items():
                    sparse_representation.update_values(
                        sympy_to_opers_str_list, old_value, new_value
                    )

    def __setitem__(self, indexing, value: [int, float, complex]):
        if isinstance(indexing, self.__class__):
            old_val, str_list = self._set_opersequence(indexing, value)
        else:
            handler_func_name = self._setitem_handlers.get(
                type(indexing), "_handle_set_default"
            )
            handler_func = getattr(self, handler_func_name)
            old_val, str_list = handler_func(indexing, value)
        if hasattr(self, "basis_dict"):
            for fock_basis, sparse_representation in self.basis_dict.items():
                sparse_representation.update_values(str_list, old_val, value)

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
                str_list.append(self.oper_list_to_str(check_item))
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

    #####################
    ### ACT ON STATES ###
    #####################
    def act_oper(self, oper: int, states: np.ndarray):
        """
        Applies an operator to a list of states, to generate the new states

        Args:
            oper (int): integer representing an operator
            states (ndarray): list of integers representing Fock states
        Returns:
            states_alive_filt: a boolean filter signifying which states survived
            new_states: list of the states after applying the operator
            signs: list of signs resulting from applying the operator
        """
        ## Check the creation/annihilation bit
        check_bit = oper & 0b1

        ## Get the position of the bit to flip
        flip_bit_pos = oper >> 1
        flip_bit = 1 << flip_bit_pos

        ## Check which states are alive
        states_alive_filt = ((states >> flip_bit_pos) & 1) == check_bit
        states_alive = states[states_alive_filt]

        ## Calculate relative signs of the operator
        signs = 1 - 2 * (np.bitwise_count(states_alive & (flip_bit - 1)) & 0b1).astype(
            np.int8
        )

        ## Return the filter for alive states, the new states and their signs
        return states_alive_filt, states_alive ^ flip_bit, signs

    def act_oper_list(self, oper_list: List[int], states: np.ndarray):
        """
        Apply a sequence of operators to a list of states,
        removes any 'destroyed' terms
        Args:
            oper_list: list of ints representing a sequence of states
            states: array of ints representing Fock states
        Returns:
            states: list of the original states which were not destroyed
            new_states: resulting new states
            signs: relative signs after applying the sequence of operators
        """
        new_states = states
        for idx, oper in enumerate(oper_list):
            states_alive_filt, new_states, new_parity = self.act_oper(oper, new_states)
            if idx == 0:
                signs = new_parity
            else:
                signs = signs[states_alive_filt] * new_parity
            states = states[states_alive_filt]
        return states, new_states, signs

    def bra_oper_ket(self, states: list, phi: np.ndarray, operators: list):
        """
        Calculate <phi|operator|phi> matrix for an operator sequence
        If multiple operator sequences are included, the contributions will be summed
        Args:
            states: list of states to operate on
            phi: array of weights of the states
            operators: list of operator lists to apply
        """
        size = len(phi[:, 0])
        T_ij = np.zeros((size, size), dtype=complex)
        trans_array = np.zeros((len(phi[0]), len(phi)), dtype="complex")

        for oper_list in operators:
            old_states, new_states, signs = self.act_oper_list(oper_list, states)

            col_indices = [
                np.where(states == old_states[i])[0][0]
                for i in range(len(old_states))
                if new_states[i] in states
            ]
            row_indices = [
                np.where(states == new_states[i])[0][0]
                for i in range(len(new_states))
                if new_states[i] in states
            ]
            relevant_signs = [
                [signs[i]] for i in range(len(signs)) if new_states[i] in states
            ]

            if len(row_indices) > 0:
                trans_array[row_indices] = (
                    np.transpose(phi[:, col_indices]) * relevant_signs
                )
                T_ij += np.conj(phi) @ trans_array
        return T_ij
