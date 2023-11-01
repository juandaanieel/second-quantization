# %% [markdown]
# # Alberto's project

# %%
# Manipulating data
import numpy as np
import sympy  # install as pip install git+https://github.com/sympy/sympy.git -> IMPORTANT!!
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import matplotlib.pyplot as plt

# Perturbation theory
from pymablock import block_diagonalize # to install follow https://pymablock.readthedocs.io/en/latest/#installation

# Computing matrix elements
from second_quantization import to_matrix

# %% [markdown]
# ## Hamiltonian definition

# %%
symbols = sympy.symbols(
    r"U n_C t_L t_R \phi \Gamma_L \Gamma_R \xi_L \xi_R E_L E_R",
    real=True,
    commutative=True,
    positive=True
)

U, n_C, t_L, t_R, phi, Gamma_L, Gamma_R, xi_L, xi_R, E_L, E_R = symbols
t_L_complex = sympy.symbols(r"t_Lc", real=False, commutative=True) # will replace t_L * exp(I * phi)

# %%
# Define 2nd quantization operators

# Dot operators
c_up, c_down = FermionOp('c_{\\uparrow}'), FermionOp('c_{\\downarrow}')

# Superconductor operators
d_L_up, d_L_down = FermionOp('d_{L, \\uparrow}'), FermionOp('d_{L, \\downarrow}')
d_R_up, d_R_down = FermionOp('d_{R, \\uparrow}'), FermionOp('d_{R, \\downarrow}')

# ABS operators, will do Bogoliubov transformation
f_L_up, f_L_down = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{L, \\downarrow}')
f_R_up, f_R_down = FermionOp('f_{R, \\uparrow}'), FermionOp('f_{R, \\downarrow}')

# Occupation operators
n = Dagger(c_up) * c_up + Dagger(c_down) * c_down
n_L = (Dagger(d_L_up) * d_L_up + Dagger(d_L_down) * d_L_down)
n_R = (Dagger(d_R_up) * d_R_up + Dagger(d_R_down) * d_R_down)

# %%
# H_abs_L_2 = xi_L * n_L + Gamma_L * Dagger(d_L_up) * Dagger(d_L_down) + Dagger(Gamma_L * Dagger(d_L_up) * Dagger(d_L_down))
# H_abs_R_2 = xi_R * n_R + Gamma_R * Dagger(d_R_up) * Dagger(d_R_down) + Dagger(Gamma_R * Dagger(d_R_up) * Dagger(d_R_down))
# H_abs_2 = H_abs_L_2 + H_abs_R_2
H_abs_L = xi_L - E_L + E_L * Dagger(f_L_up) * f_L_up + E_L * Dagger(f_L_down) * f_L_down
H_abs_R = xi_R - E_R + E_R * Dagger(f_R_up) * f_R_up + E_R * Dagger(f_R_down) * f_R_down
H_abs = H_abs_L + H_abs_R
display(sympy.Eq(sympy.Symbol('H_{ABS}'), H_abs))
print(' ')

H_dot = U * (n - n_C)**2 / 2
display(sympy.Eq(sympy.Symbol('H_{dot}'), H_dot))
print(' ')

H_TL = t_L_complex * (Dagger(c_up) * d_L_up + Dagger(c_down) * d_L_down)
H_TR = t_R * (Dagger(c_up) * d_R_up + Dagger(c_down) * d_R_down)
H_T = H_TL + H_TR + Dagger(H_TL + H_TR)
display(sympy.Eq(sympy.Symbol('H_{T}'), H_T))

H = H_abs + H_dot + H_T
# %%
# Bogoliubov coefficients
u_L , v_L, u_R, v_R = sympy.symbols(
    r"u_L v_L u_R v_R",
    real=True,
    commutative=True,
    positive=True
)

# Bogoliubov transformation
transformation_substitution = {
    d_L_down: u_L * f_L_down + v_L * Dagger(f_L_up),
    d_L_up: u_L * f_L_up - v_L * Dagger(f_L_down),
    d_R_down: u_R * f_R_down + v_R * Dagger(f_R_up),
    d_R_up: u_R * f_R_up - v_R * Dagger(f_R_down),
}
transformation_substitution.update({Dagger(key): Dagger(value) for key, value in transformation_substitution.items()})
H = H.subs(transformation_substitution, simultaneous=True).expand()

# %%
%%time
H_matrix, basis = to_matrix(H)

# %%
H_matrix = H_matrix.subs({t_L_complex: t_L * sympy.cos(phi) + t_L * sympy.I * sympy.sin(phi)})
# %% [markdown]
# ## Perturbation theory
#
# First, we define a ground state for each $n$ occupation number of the quantum dot.
# Then, we compute the effective Hamiltonian for each case.

# %%
# To get some idea for what is the ground state for different parameters.
params = {n_C:1, U:10, xi_L:0, xi_R:0, E_L: 1, E_R: 1}
evals_H0 = np.array(H_matrix.diagonal().subs(params), dtype=float)[0]

print(f'Ground state energy is {np.min(evals_H0)} for states:')
for index in np.where(evals_H0 == np.min(evals_H0))[0]:
    display(basis[index])

# %%
# Possible ground states
# For n=0
ground_state_n0 = [sympy.S.One]
# For n=1
ground_state_n1 = [c_up, c_down] 
# For n=2
ground_state_n2 = [c_up * c_down]
# %%
# Choose ground state
ground_state = ground_state_n2

print(f'Ground state is: ')
for element in ground_state:
    display(element)
subspace_indices = [int(not(element in ground_state)) for element in basis]
# %%
%%time
H_tilde = block_diagonalize(H_matrix, subspace_indices=subspace_indices, symbols=[t_L, t_R])[0]
# %%
%%time
correction = H_tilde[0, 0, 2, 2]
# %%
%%time
trace = sympy.Add(*[correction[i, i].expand() for i in range(correction.shape[0])])
current = sympy.diff(trace, phi)
prefactor = Gamma_L * Gamma_R * t_L ** 2 * t_R ** 2 / (E_L * E_R) * 2 * sympy.sin(phi) * sympy.cos(phi)
result = (current / prefactor).expand()
# %%
result

# %%
# Express Bogoliubov coefficients as functions of Gamma and E, avoiding square roots

alpha_L, alpha_R  = sympy.symbols('alpha_L alpha_R', real=True, positive=True, commutative=True)

u_L_subs = sympy.cos(alpha_L)
v_L_subs = sympy.sin(alpha_L)

u_R_subs = sympy.cos(alpha_R)
v_R_subs = sympy.sin(alpha_R)

uv_subs = {u_L * v_L : Gamma_L / (2 * E_L), u_R * v_R : Gamma_R / (2 * E_R)}


# %%
def simplify_fraction(expr):
    expr = expr.expand()
    result = sympy.Add(*[term.factor() for term in expr.as_ordered_terms()])

    # Group fractions with common denominator
    fractions = {}
    for term in result.as_ordered_terms():
        numerator, denominator = sympy.fraction(term)
        if denominator in fractions:
            fractions[denominator] += numerator
        else:
            fractions[denominator] = numerator

    # Put sum back together after simplifying numerator
    result = sum([(numerator.subs(uv_subs) / denominator) for denominator, numerator in fractions.items()])
    return result


# %%
simplified_result = simplify_fraction(result)

# %%
# I_0
simplified_result

# %%
prefactor

# %% [markdown]
# ## Replacing numerical values

# %%
numerical_supercurrent = simplified_result * prefactor

U_value = 10
Gamma_L_value = 0.5
Gamma_R_value = 0.5
xi_L_value = 0.3 * Gamma_L_value
xi_R_value = 0
t_L_value = 0.3 * Gamma_L_value
t_R_value = 0.3 * Gamma_R_value
E_R_value = np.sqrt(Gamma_R_value**2 + xi_R_value**2)
E_L_value = np.sqrt(Gamma_L_value**2 + xi_L_value**2)

values = {
    U: U_value,
    Gamma_L: Gamma_L_value,
    Gamma_R: Gamma_R_value,
    t_L: t_L_value,
    t_R: t_R_value,
    E_R: E_R_value,
    E_L: E_L_value,
    phi: np.pi/4}

n_C_values = np.linspace(0, 4, 100)
supercurrent_values = []
for n_C_value in n_C_values:
    values.update({n_C: n_C_value})
    supercurrent_values.append(numerical_supercurrent.subs(values))

supercurrent_values = np.array(supercurrent_values)

# %%
plt.plot(n_C_values[1:], supercurrent_values[1:])
# plt.ylim((-0.6, 3))

# %% [markdown]
# ## Results alberto had up to 4th order

# %% [markdown]
# ![image.png](attachment:093b5f9c-4880-4d05-adc8-6d0fa5ad2607.png)

# %%
E_0 = U * n_C**2 / 2
E_1 = (U * (1 - n_C)**2 / 2).expand()
E_2 = (U * (2 - n_C)**2 / 2).expand()

I_0 = 4/ ((E_1 - E_0 + E_L) * (E_2 - E_0) * (E_1 - E_0 + E_R)) + 2 / ((E_1 - E_0 + E_L) * (E_L + E_R) * (E_1 - E_0 + E_R))
display(sympy.Eq(sympy.Symbol('I_{c}(n=0)'), I_0))

I_1 = 1 / ((E_0 - E_1 + E_L) * (E_L + E_R) * (E_0 - E_1 + E_R)) + 1 / ((E_2 - E_1 + E_L) * (E_L + E_R) * (E_2 - E_1 + E_R))
display(sympy.Eq(sympy.Symbol('I_{c}(n=1)'), I_1))

I_2 = 4 / ((E_1 - E_2 + E_L) * (E_0 - E_2) * (E_1 - E_2 + E_R)) + 2 / ((E_1 - E_2 + E_L) * (E_L + E_R) * (E_1 - E_2 + E_R))
display(sympy.Eq(sympy.Symbol('I_{c}(n=2)'), I_2))

# %%
I_0.expand().factor()

# %%
I_1.expand().factor()

# %%
simplified_result.expand().factor()
