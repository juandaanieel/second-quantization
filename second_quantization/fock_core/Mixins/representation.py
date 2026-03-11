from typing import Callable, List, Any, Self
import sympy 
import numpy as np

#### Store functionality for markdown printing of the FermionSequence class
class RepresentationMixin:
    def __init__(self):
        return

    #####################
    ## Format Markdown ##
    #####################
    @staticmethod
    def _format_weight_for_repr(w: complex) -> str :
        if isinstance(w, sympy.Basic):
            return " $+$ "+f'({w._repr_latex_()})'
        
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

    def vis_oper_list(self, oper_list:List[List[int]], displ: bool=False):
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
    