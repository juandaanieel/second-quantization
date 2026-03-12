from second_quantization.fock_core.fermion_sequence import FermionSequence

__all__ = ["FermionSequence"]


### TODO
### Define convenience functions that act similar to hilbert_space for later compatibility
def make_system():
    return


def to_matrix():
    return


def make_dict_callable():
    return


I = FermionSequence(1)  # noqa: E741
eye = I
Zero = FermionSequence()
c_down = FermionSequence([0])
c_up = FermionSequence([2])
a_up = ~c_up
a_down = ~c_down
