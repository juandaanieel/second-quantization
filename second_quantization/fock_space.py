from second_quantization.fock_core.fermion_sequence import FermionSequence
from second_quantization.fock_core.fock_basis import FockBasis

__all__ = ["FermionSequence", "FockBasis"]

I = FermionSequence(1)  # noqa: E741
eye = I
Zero = FermionSequence()
c_down = FermionSequence([0])
c_up = FermionSequence([2])
a_up = ~c_up
a_down = ~c_down
