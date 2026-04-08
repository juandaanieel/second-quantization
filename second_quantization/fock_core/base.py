from typing import List, Callable
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.fermion import FermionOp

import numpy as np


class FermionSequenceBase:
    ##################
    ## Parse Inputs ##
    ##################
    _parseinput_handlers = {
        int: "_parse_int",
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
    def _construct_operator(position: int, spin: int, annihilation: bool = True):
        """
        Creates the integer representing an operator, in the default normal ordered convention.
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

    @staticmethod
    def _operator_verbose(type_str: str, position: int, spin: str):
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
        oper = FermionSequenceBase._construct_operator(position, spin, annihilation)
        return oper

    def _parse_sympy(
        self, sympy_expr, sympy_operator_order: List[FermionOp]
    ) -> List[List]:
        if sympy_operator_order is None:
            raise ValueError(
                "SymPy parsing requires `sympy_operator_order` to be provided."
            )
        sympy_expr = sympy_expr.expand()
        oper_list, weights = [], []

        for term in sympy_expr.as_ordered_terms():
            sublist = []
            coeff = 1
            for factor in term.as_ordered_factors():
                if factor in sympy_operator_order:
                    sublist.append(1 + 2 * sympy_operator_order.index(factor))
                elif Dagger(factor) in sympy_operator_order:
                    sublist.append(0 + 2 * sympy_operator_order.index(Dagger(factor)))
                elif isinstance(factor, sympy.Symbol):
                    coeff *= factor
                else:
                    coeff *= factor

            sublist.reverse()
            oper_list.append(sublist)
            weights.append(coeff)

        return oper_list, weights

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
                output.append(FermionSequenceBase._operator_verbose(*subop_tuple))
            output.reverse()
            return output
        else:
            return [FermionSequenceBase._operator_verbose(*operator_tuple)]

    @staticmethod
    def oper_list_to_key(oper_list):
        """
        Convert a list of integers representing operators
        to an immutable dictionary key representation.
        """
        return tuple(oper_list)

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
        if len(oper_list) == 0:
            return states, states, np.ones(len(states), dtype=np.int8)

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

    def apply_to_weight(self, func: Callable, *args, **kwargs):
        for idx, w in enumerate(self.weights):
            self.weights[idx] = func(w, *args, **kwargs)
        return self
