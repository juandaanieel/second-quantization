# +
import numpy as np
from second_quantization.fock_space import FermionSequence, c_down, c_up, a_down, a_up

def build_H_fermi_hubbard(N):
    ## Make ECT terms
    ECT = (c_down*(~c_down>>1) + c_up*(~c_up>>1)) >> np.arange(1,N-1)

    MU = (c_down*(~c_down) + c_up*(~c_up)) >> np.arange(1,N)
    
    U = (c_up*c_down*a_up*a_down) >> np.arange(1,N)
        
    return U, MU, ECT


# -

N=5
U,MU,ECT = build_H_fermi_hubbard(N)
H = U + MU + ECT
H

## Currently works with sympy symbols as weights but not yet in the same structure as hilbert_space
## ToDo - implement the same FermionOp instantiation
## Also - fix printing in a nicer way (or convert to SymPy before printing)
import sympy as sp
u, t, mu = sp.symbols('U, t,mu')
H_symb = u*U + mu*MU + t*ECT
H_symb

sparse_data = H.to_sparse(N)
coo_sparse = sparse_data.to_sparse_coo()
csr_sparse = sparse_data.to_sparse_csr()
np_array = sparse_data.to_array()


