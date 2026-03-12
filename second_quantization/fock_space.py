from second_quantization.fock_core.fermion_sequence import FermionSequence
from sympy.physics.quantum import Dagger

__all__ = ["FermionSequence"]


def to_sequence(H, operators):
    H = H.expand()

    oper_list = []
    weights = []

    for term in H.as_ordered_terms():
        sublist = []
        coeff = 1
        for factor in term.as_ordered_factors():
            if factor in operators:
                sublist.append(1 + 2 * operators.index(factor))
            elif Dagger(factor) in operators:
                sublist.append(0 + 2 * operators.index(Dagger(factor)))
            else:
                coeff *= factor
        sublist.reverse()
        oper_list.append(sublist)
        weights.append(coeff)
    return FermionSequence(
        oper_list,
        weights=weights,
        bypass_parse=True,
    )


I = FermionSequence(1)  # noqa: E741
eye = I
Zero = FermionSequence()
c_down = FermionSequence([0])
c_up = FermionSequence([2])
a_up = ~c_up
a_down = ~c_down
