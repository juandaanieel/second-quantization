import numpy as np
import sympy
from typing import List, Self
from copy import deepcopy
from collections import Counter


class FermionAlgebraMixin:
    """
    Mixin implementing algebra, normalization, and comparison behavior for
    fermionic operator sequences.

    Expects the host class to provide:
    - `self.oper_list`
    - `self.weights`
    - construction via `self.__class__(..., bypass_parse=True)`
    """

    ######################################################
    #####             OPERATOR ARITHMETIC            #####
    ######################################################

    ## Specify the non-self.__class__ types that are allowed to be added/multiplied to an FermionSequence
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

        result = self.__class__(new_oper_list, weights=new_weights, bypass_parse=True)
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

        result = self.__class__(new_oper_list, weights=new_weights, bypass_parse=True)
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
        negative_oper = self.__class__(
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

    def __mul__(self, multiplier: [int, float, complex]):
        if not isinstance(multiplier, (self.__class__, *self._allowed_types)):
            raise TypeError(
                f"Unsupported operand type(s) for *: {self.__class__.__name__} and '{type(multiplier).__name__}'"
            )

        ### For multiplying by a scalar/Symbol, multiply the weights and return a new FermionSequence
        if not isinstance(multiplier, self.__class__):
            new_weights = [w * multiplier for w in self.weights]
            product = self.__class__(
                self.oper_list, weights=new_weights, bypass_parse=True
            )
            return product

        ### Otherwise, multiply sequences together and clean the result
        product_outcome = self._multiplication_basis(
            self.weights, self.oper_list, multiplier.weights, multiplier.oper_list
        )
        product = self.__class__(
            product_outcome[1], weights=product_outcome[0], bypass_parse=True
        )
        product.clean()
        return product

    def __rmul__(self, multiplier):
        if not isinstance(multiplier, self._allowed_types):
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(multiplier).__name__}' and {self.__class__.__name__}"
            )
        new_weights = [w * multiplier for w in self.weights]
        product = self.__class__(self.oper_list, weights=new_weights, bypass_parse=True)
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

    def __truediv__(self, divisor: [int, float]):
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
        return self.__class__(oper_list, weights=new_weights, bypass_parse=True)

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

    def __invert__(self):
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

        inverse = self.__class__(
            invert_list,
            weights=[
                w.conjugate() if isinstance(w, (complex, sympy.Basic)) else w
                for w in self.weights
            ],
            bypass_parse=True,
        )
        return inverse

    dag = __invert__  ## Alias for complex conjugation

    def __rshift__(self, shift_by: int):
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
            shifted_sequence = self.__class__(
                new_opers, weights=self.weights[:], bypass_parse=True
            )
            return shifted_sequence
        else:
            base_sequence = self.__class__(
                deepcopy(self.oper_list), weights=self.weights[:], bypass_parse=True
            )
            shift_sequence = self.__class__(
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

    def __lshift__(self, shift_by: int):
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
        shifted_sequence = self.__class__(
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

    def __eq__(self, oper_sequence) -> bool:
        """
        Check equality of two FermionSequences.
        Two sequences are determined equal if they have identical subsequences with the same weights,
        regardless of the order they appear in.
        """
        if not isinstance(oper_sequence, self.__class__):
            return False
        return Counter(self._normalized()) == Counter(oper_sequence._normalized())

    def __ne__(self, oper_sequence) -> bool:
        """
        Check inequality of two FermionSequence.
        Two sequences are determined non-equal if they do not have identical subsequences,
        regardless of the order they appear in.
        """
        if not isinstance(oper_sequence, self.__class__):
            return True
        return Counter(self._normalized()) != Counter(oper_sequence.normalized())
