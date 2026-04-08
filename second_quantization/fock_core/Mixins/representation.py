from typing import List
import sympy
import numpy as np
from IPython.core.display import Markdown
from IPython.display import display


class RepresentFermionsMixin:
    """
    Mixin implementing human-readable and markdown representations for
    fermionic operator sequences.

    Expects the host class to provide:
    - `self.oper_list`
    - `self.weights`
    - optionally `self.to_sympy()`
    - optionally `self._sympy_operators`
    """

    #####################
    ## Format Markdown ##
    #####################
    @staticmethod
    def _format_weight_for_repr(w: complex) -> str:
        if isinstance(w, sympy.Basic):
            return " $+$ " + f"({w._repr_latex_()})"

        w_str = ""
        ## w purely real
        if w.imag == 0 and w.real != 0:
            if w.real > 0:
                w_str += " $+$ "
            else:
                w_str += " $-$ "
            if abs(w) == 1:
                return w_str
            else:
                w_str += f"{abs(np.round(w,3))}"
                return w_str

        ## w purely imaginary
        elif w.real == 0 and w.imag != 0:
            if w.imag > 0:
                w_str += " $+$ "
            else:
                w_str += " $-$ "
            if abs(w) == 1:
                w_str += "j"
                return w_str
            else:
                w_str += f"{abs(np.round(w,3))}j"
                return w_str

        ## w is a mix
        w_str += " $+$ " + f"{np.round(w,3)}"
        return w_str

    @staticmethod
    def vis_oper(oper: int):
        """
        Convert an integer representing an operator to a readable string
        Args:
            oper (int): integer representing an operator
        Returns:
            (str): string visualizing the operator

        """
        spins = ["\u2193", "\u2191"]

        creation = (oper & 0b1) == 0
        act_pos = oper >> 1
        act_pos_proper = np.floor(act_pos / 2)
        spin_vis = spins[act_pos % 2]

        if creation:
            return "$c^{\u2020}_{" + f"{int(act_pos_proper)},{spin_vis}" + "}$"
        else:
            return "$c_{" + f"{int(act_pos_proper)},{spin_vis}" + "}$"

    def vis_oper_list(self, oper_list: List[List[int]], displ: bool = False):
        """
        Convert a sequence of operators to a readable string
        """
        if len(oper_list) == 0:
            return "$\\hat{I}$"
        full_str = ""
        for oper in list(reversed(oper_list)):
            full_str += self.vis_oper(oper)
        if displ:
            display(Markdown(full_str))
        else:
            return full_str

    def _repr_markdown_(self):
        if hasattr(self, "_sympy_operators"):
            return self.to_sympy()[0]._repr_latex_()

        operstr = ""
        ## Return Zero symbol for empty operator
        if len(self.oper_list) == 0:
            return "$\\hat{0}$"

        ## Otherwise build the string by iterating through the subsequences and weights
        for w, oper_seq in zip(self.weights, self.oper_list):
            operstr += self._format_weight_for_repr(w) + self.vis_oper_list(oper_seq)

        ## Remove the first " $+$ " for formatting purposes
        if operstr[0:4] == " $+$":
            return operstr[4:]
        return operstr

    def __str__(self):
        return self._repr_markdown_()


class RepresentFockBasisMixin:
    """
    Mixin implementing human-readable and markdown representations for Fock bases.

    Expects the host class to provide:
    - `self.states`
    - `self.N`
    """

    def vis_state(self, state):
        """
        Show a 'state' in a readable fashion
        args:
            state (int): the Fock state to visualize
            displ (bool): if True, prints the state
                          if False, returns a string
        """
        n = state
        if isinstance(n, np.ndarray):
            if len(n) == 0:
                return "0"
            elif n[0] == -1:
                return "0"
        elif n < 0:
            return "0"

        state_str = "\u007c"
        count = 0
        while n:
            check_up = n & 1
            n >>= 1
            check_down = n & 1
            n >>= 1
            if check_up and check_down:
                state_str += "\u2193\u2191,"
            elif check_up:
                state_str += "\u2193,"
            elif check_down:
                state_str += "\u2191,"
            else:
                state_str += "0,"
            count += 1

        while count < self.N / 2:
            state_str += "0,"
            count += 1
        state_str = state_str[:-1]
        state_str += "\u3009"
        return state_str

    def vis_state_list(
        self,
        states,
    ):
        """
        Visualise a superposition of Fock states
        Args:
            states (list): list of integers representign states
            weights (floats): the weight of each state
            displ (bool): if True prints the visualisation
                          if False returns the string
        """
        state_str = ""
        for state in states:
            state_str += self.vis_state(state) + " "
        return state_str

    def _repr_markdown_(self) -> str:
        info_string = f"Fock basis with {len(self.states)} states<br>"
        if len(self.states) > 32:
            return (
                info_string
                + self.vis_state_list(self.states[:5])
                + "$\\cdots$ "
                + self.vis_state_list(self.states[-5:])
            )
        return info_string + self.vis_state_list(self.states)
