# %%
# Manipulating data
import numpy as np
import sympy  # install as pip install git+https://github.com/sympy/sympy.git -> IMPORTANT!!
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger

# Perturbation theory
from pymablock import block_diagonalize # to install follow https://pymablock.readthedocs.io/en/latest/#installation

# Computing matrix elements
from second_quantization import to_matrix

# %%
symbols = sympy.symbols(
    r"U n_C t_L t_R \phi \Gamma_L \Gamma_R E_L E_R",
    real=True,
    commutative=True,
    positive=True
)

U, n_C, t_L, t_R, phi, Gamma_L, Gamma_R, E_L, E_R = symbols

# %%
# Define 2nd quantization operators

# Dot operators
c_up, c_down = FermionOp('c_{\\uparrow}'), FermionOp('c_{\\downarrow}')

# Superconductor operators
d_L_up, d_L_down = FermionOp('d_{L, \\uparrow}'), FermionOp('d_{L, \\downarrow}')
d_R_up, d_R_down = FermionOp('d_{R, \\uparrow}'), FermionOp('d_{R, \\downarrow}')

# Occupation operators
n = Dagger(c_up) * c_up + Dagger(c_down) * c_down
n_L = (Dagger(d_L_up) * d_L_up + Dagger(d_L_down) * d_L_down)
n_R = (Dagger(d_R_up) * d_R_up + Dagger(d_R_down) * d_R_down)

# %%
# Define Hamiltonians
onsite_abs = E_L * n_L + E_R * n_R
gammas_abs = Gamma_L * Dagger(d_L_up) * Dagger(d_L_down) + Gamma_R * Dagger(d_R_up) * Dagger(d_R_down)
H_abs =  onsite_abs + gammas_abs + Dagger(gammas_abs)
display(sympy.Eq(sympy.Symbol('H_{ABS}'), H_abs))
print(' ')

H_dot = U * (n - n_C)**2 / 2
display(sympy.Eq(sympy.Symbol('H_{dot}'), H_dot))
print(' ')

H_TL = t_L * sympy.exp(sympy.I * phi) * (Dagger(c_up) * d_L_up + Dagger(c_down) * d_L_down)
H_TR = t_R * (Dagger(c_up) * d_R_up + Dagger(c_down) * d_R_down)
H_T = H_TL + H_TR + Dagger(H_TL + H_TR)
display(sympy.Eq(sympy.Symbol('H_{T}'), H_T))

H = H_abs + H_dot + H_T
# %%
%%time
H_matrix, basis = to_matrix(H)
H_matrix = H_matrix.rewrite(sympy.cos)
assert H_matrix.is_hermitian
# %%
# Choose the ground state

# For n=0
occ_subspace = [sympy.S.One]
# For n=1
# ???
# For n=2
# occ_subspace = [c_up, c_down]

subspace_indices = [int(not(element in occ_subspace)) for element in basis]
# %%
%%time
H_tilde = block_diagonalize(H_matrix, subspace_indices=subspace_indices, symbols=[t_L, t_R, Gamma_L, Gamma_R], algorithm='expanded')[0]
# %%
%%time
result = H_tilde[0, 0, 2, 2, 1, 1]
display(result)
# %%
%%time
trace = sum([result[i, i] for i in range(result.shape[0])])
# %%
current = sympy.diff(trace, phi).expand()
prefactor = Gamma_L * Gamma_R * t_L ** 2 * t_R ** 2 / (E_L * E_R) * 2 * sympy.sin(phi) * sympy.cos(phi)
result = (current / prefactor).expand()
# %%
result = result.together().factor()
# %%
result
# %%
