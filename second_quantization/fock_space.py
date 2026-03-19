from second_quantization.fock_core.fermion_sequence import FermionSequence
from second_quantization.fock_core.fock_basis import FockBasis

__all__ = ["FermionSequence", "FockBasis"]

"""
def to_sequence(H, operators):
    H = H.expand()

    oper_list = []
    weights = []
    sympy_dict = {}

    for term in H.as_ordered_terms():
        sublist = []
        coeff = 1
        sympy_key = 1
        for factor in term.as_ordered_factors():
            if factor in operators:
                sublist.append(1 + 2 * operators.index(factor))
            elif Dagger(factor) in operators:
                sublist.append(0 + 2 * operators.index(Dagger(factor)))
            elif isinstance(factor, sympy.Symbol):
                coeff *= factor
                sympy_key *= factor
            else:
                coeff *= factor

        sublist.reverse()
        oper_list.append(sublist)
        weights.append(coeff)
        if sympy_key not in sympy_dict.keys():
            sympy_dict[sympy_key] = (0, [])
        sympy_dict[sympy_key][1].append(FermionSequence.oper_list_to_str(sublist))

    ferm_sequence = FermionSequence(oper_list, weights=weights, bypass_parse=True)
    ferm_sequence._sympy_operators = operators
    ferm_sequence._sympy_dict = sympy_dict

    return ferm_sequence
"""

I = FermionSequence(1)  # noqa: E741
eye = I
Zero = FermionSequence()
c_down = FermionSequence([0])
c_up = FermionSequence([2])
a_up = ~c_up
a_down = ~c_down
