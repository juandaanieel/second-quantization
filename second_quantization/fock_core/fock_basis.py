import numpy as np
from .Mixins.representation import RepresentFockBasisMixin
from typing import List


class FockBasis(RepresentFockBasisMixin):
    """
    Representation class for a Fock basis that allows to construct Hamiltonians in unordered or symmetry reduced subspaces.
    A Fock basis consists of a list of integers, which in binary indicate whether a Fermionic mode is occupied (bit 1) or unoccupied (bit 0)
    """

    def __init__(self, states: [int, np.ndarray], N: int = None):
        self.states = self._parse_states(states)
        ## Ideally one does not calculate the hash dictionary if the fock states are fock ordered
        if N is None:
            N = int(np.ceil(np.log2(np.max(self.states) + 1)))
        self.N = N
        self.is_ordered = True if len(self.states) == 2**N else False
        self.state_to_index = None
        self.state_index_array = None
        if not self.is_ordered:
            self.state_to_index = {num: idx for idx, num in enumerate(self.states)}
            self._build_state_index_array()
        self.hashed = self.state_to_index

    def __len__(self):
        return len(self.states)

    def _build_state_index_array(self):
        max_state = int(np.max(self.states))
        self.state_index_array = np.full(max_state + 1, -1, dtype=np.int32)
        self.state_index_array[self.states] = np.arange(
            len(self.states), dtype=np.int32
        )

    def get_index(self, state: int):
        if self.is_ordered:
            return state if 0 <= state < len(self.states) else None
        if state < 0:
            return None
        if self.state_index_array is not None and state < len(self.state_index_array):
            index = int(self.state_index_array[state])
            return index if index >= 0 else None
        return self.state_to_index.get(state)

    def get_indices(self, states: np.ndarray):
        if self.is_ordered:
            in_basis = (states >= 0) & (states < len(self.states))
            return states[in_basis], in_basis

        if self.state_index_array is not None:
            in_range = (states >= 0) & (states < len(self.state_index_array))
            indices = np.full(len(states), -1, dtype=np.int32)
            indices[in_range] = self.state_index_array[states[in_range]]
            in_basis = indices >= 0
            return indices[in_basis], in_basis

        indices = np.array(
            [self.state_to_index.get(state, -1) for state in states], dtype=np.int32
        )
        in_basis = indices >= 0
        return indices[in_basis], in_basis

    @staticmethod
    def _parse_states(states: [int, List, np.ndarray]):
        if isinstance(states, int):
            return np.arange(2**states, dtype=np.int32)
        if isinstance(states, list):
            return np.array(states)
        if isinstance(states, np.ndarray):
            return states

    def restrict(self, *masks: "Mask") -> "FockBasis":
        states = self.states
        combined_mask = np.ones(len(states), dtype=bool)

        for mask in masks:
            combined_mask &= mask.apply(states, self.N)

        return FockBasis(states[combined_mask], N=self.N)


class Mask:
    def apply(self, states: np.ndarray, N: int) -> np.ndarray:
        """
        Returns a boolean mask of same length as states.
        """
        raise NotImplementedError


class ParityMask(Mask):
    def __init__(self, parity: str):
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        self.parity = parity

    def apply(self, states, N):
        even_mask = np.bitwise_count(states) % 2 == 0
        return even_mask if self.parity == "even" else ~even_mask


class SpinMask(Mask):
    def __init__(self, allowed: str):
        self.allowed = allowed

    def apply(self, states, N):
        mask = np.ones(len(states), dtype=bool)
        for n in range(N):
            down = 1 << (2 * n)
            up = 1 << (2 * n + 1)

            if self.allowed == "down":
                mask &= (states & up) == 0
            elif self.allowed == "up":
                mask &= (states & down) == 0
        return mask


class DoubleOccupancyMask(Mask):
    def apply(self, states, N):
        mask = np.ones(len(states), dtype=bool)
        for n in range(N):
            down = 1 << (2 * n)
            up = 1 << (2 * n + 1)
            both = ((states & down) != 0) & ((states & up) != 0)
            mask &= ~both
        return mask
