# %% [markdown]
# # Alberto's project

# %%
# Manipulating data
import numpy as np
import sympy  # install as pip install git+https://github.com/sympy/sympy.git -> IMPORTANT!!
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
import matplotlib
import matplotlib.pyplot as plt

# Perturbation theory
from pymablock import block_diagonalize # to install follow https://pymablock.readthedocs.io/en/latest/#installation

# Computing matrix elements
from second_quantization import to_matrix

# %%
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.labelsize"] = 14

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

ts = t_L_complex, t_R
Gammas = Gamma_L, Gamma_R
xis = xi_L, xi_R
Es = E_L, E_R

# %%
# Define 2nd quantization operators

# Dot operators
c_up, c_down = FermionOp('c_{\\uparrow}'), FermionOp('c_{\\downarrow}')

# Superconductor operators
d_ups = FermionOp('d_{L, \\uparrow}'), FermionOp('d_{R, \\uparrow}')
d_downs = FermionOp('d_{L, \\downarrow}'), FermionOp('d_{R, \\downarrow}')

# ABS operators, will do Bogoliubov transformation
f_ups = FermionOp('f_{L, \\uparrow}'), FermionOp('f_{R, \\uparrow}')
f_downs = FermionOp('f_{L, \\downarrow}'), FermionOp('f_{R, \\downarrow}')

# Occupation operators
n = Dagger(c_up) * c_up + Dagger(c_down) * c_down

# %%
# Define Hamiltonians

H_abs = 0
for xi, E, f_up, f_down in zip(xis, Es, f_ups, f_downs):
    H_abs += xi - E + E * Dagger(f_up) * f_up + E * Dagger(f_down) * f_down
display(sympy.Eq(sympy.Symbol('H_{ABS}'), H_abs))
print(' ')

H_dot = U * (n - n_C)**2 / 2
display(sympy.Eq(sympy.Symbol('H_{dot}'), H_dot))
print(' ')

H_T = 0
for t, d_up, d_down in zip(ts, d_ups, d_downs):
    H_T += t * (Dagger(c_up) * d_up + Dagger(c_down) * d_down)
H_T += Dagger(H_T)
display(sympy.Eq(sympy.Symbol('H_{T}'), H_T))

H = H_abs + H_dot + H_T
# %%
# Bogoliubov coefficients
us = u_L, u_R = sympy.symbols(r"u_L u_R", real=True, commutative=True, positive=True)
vs = v_L, v_R = sympy.symbols(r"v_L v_R", real=True, commutative=True, positive=True)

# Bogoliubov transformation
transformation_substitution = {d_down: u * f_down + v * Dagger(f_up)
    for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups)}

transformation_substitution.update({d_up: u * f_up - v * Dagger(f_down)
    for u, v, d_down, d_up, f_down, f_up in zip(us, vs, d_downs, d_ups, f_downs, f_ups)})

transformation_substitution.update({Dagger(key): Dagger(value) for key, value in transformation_substitution.items()})

# Substitute coefficients so that H_T is in ABS basis
H = H.subs(transformation_substitution).expand()

# %%
%%time
H_matrix, basis = to_matrix(H)

# %%
H_matrix = H_matrix.subs({t_L_complex: t_L * sympy.cos(phi) + t_L * sympy.I * sympy.sin(phi)})
# %%
E_0, E_1, E_2 = sympy.symbols(r"E_0 E_1 E_2", real=True, commutative=True, positive=True)

E_0_value = U * n_C**2 / 2
E_1_value = (U * (1 - n_C)**2 / 2).expand()
E_2_value = (U * (2 - n_C)**2 / 2).expand()

H_matrix = H_matrix.subs({E_2_value: E_2})
H_matrix = H_matrix.subs({E_1_value: E_1})
H_matrix = H_matrix.subs({E_0_value: E_0})

# %% [markdown]
# ## Perturbation theory
#
# First, we define a ground state for each $n$ occupation number of the quantum dot.
# Then, we compute the effective Hamiltonian for each case.

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
ground_state = ground_state_n0

print(f'Ground state is: ')
for element in ground_state:
    display(element)
subspace_indices = [int(not(element in ground_state)) for element in basis]
# %%
%%time
H_tilde = block_diagonalize(H_matrix, subspace_indices=subspace_indices, symbols=[t_L, t_R])[0]
# %%
%%time
order_L = 2
order_R = 2
correction = H_tilde[0, 0, order_L, order_R]
# %%
%%time
trace = sympy.Add(*[correction[i, i].expand() for i in range(correction.shape[0])])
current = sympy.diff(trace, phi)
prefactor = Gamma_L * Gamma_R * t_L ** order_L * t_R ** order_R / (E_L * E_R) * 2 * sympy.sin(phi) * sympy.cos(phi)
result = (current / prefactor).expand()
# %%
# Express Bogoliubov coefficients as functions of Gamma and E, avoiding square roots

alpha_L, alpha_R  = sympy.symbols('alpha_L alpha_R', real=True, positive=True, commutative=True)

u_L_subs, u_R_subs = sympy.cos(alpha_L), sympy.cos(alpha_R) # alpha = arctan(Gamma / xi)
v_L_subs, v_R_subs = sympy.sin(alpha_L), sympy.sin(alpha_R)

uv_subs = {u * v : Gamma / (2 * E) for u, v, Gamma, E in zip(us, vs, Gammas, Es)}
u_subs = {u**2: (1 + xi / E) / 2 for u, xi, E in zip(us, xis, Es)}
v_subs = {v**2: (1 - xi / E) / 2 for v, xi, E in zip(vs, xis, Es)}


# %%
def simplify_fraction(expr):
    expr = expr.expand()
    result = sympy.Add(*[term.factor() for term in expr.as_ordered_terms()])

    # Group fractions with common denominator
    fractions = {}
    for term in result.as_ordered_terms():
        numerator, denominator = sympy.fraction(term)
        numerator = numerator.subs(uv_subs).subs(u_subs).subs(v_subs)
        if denominator in fractions:
            fractions[denominator] += numerator
        else:
            fractions[denominator] = numerator

    # Put sum back together after simplifying numerator
    result = sum([(numerator.factor() / denominator) for denominator, numerator in fractions.items()])
    return result, fractions


# %%
%%time
simplified_result, fractions = simplify_fraction(result)

# %%
simplified_result

# %%
len(fractions)

# %%
prefactor

# %% [markdown]
# ## Replacing numerical values

# %%
%%time

U_value = 5
Gamma_L_value = 1
Gamma_R_value = 1
xi_L_value = 0.3
xi_R_value = 0.3
t_L_value = 0.5
t_R_value = 0.5

values = {
    U: U_value,
    Gamma_L: Gamma_L_value,
    Gamma_R: Gamma_R_value,
    t_L: t_L_value,
    t_R: t_R_value,
    xi_L: xi_L_value,
    xi_R: xi_R_value,
    E_R: np.sqrt(Gamma_R_value**2 + xi_R_value**2),
    E_L: np.sqrt(Gamma_L_value**2 + xi_L_value**2),
    E_0: E_0_value,
    E_1: E_1_value,
    E_2: E_2_value,
    phi: np.pi/4}

numerical_prefactor = prefactor.subs(values)

n_C_values = np.linspace(0, 2, 600)
supercurrent_values = []
for n_C_value in n_C_values:
    values.update({n_C: n_C_value})
    supercurrent_values.append(simplified_result.subs(values) * numerical_prefactor)

supercurrent_values = np.array(supercurrent_values, dtype=float)

# %%
fig, ax = plt.subplots()
ax.plot(n_C_values, (supercurrent_values))
ax.set_xlabel(r'$n_C$')
ax.set_ylabel(r'$I_c$')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_ylim(-4, 4)

# %% [markdown]
# ## Results alberto had up to 4th order

# %% [markdown]
# ![image.png](attachment:093b5f9c-4880-4d05-adc8-6d0fa5ad2607.png)

# %%
I_0 = 4/ ((E_1 - E_0 + E_L) * (E_2 - E_0) * (E_1 - E_0 + E_R)) + 2 / ((E_1 - E_0 + E_L) * (E_L + E_R) * (E_1 - E_0 + E_R))
display(sympy.Eq(sympy.Symbol('I_{c}(n=0)'), I_0))

I_1 = 1 / ((E_0 - E_1 + E_L) * (E_L + E_R) * (E_0 - E_1 + E_R)) + 1 / ((E_2 - E_1 + E_L) * (E_L + E_R) * (E_2 - E_1 + E_R))
I_1 += 2 / ((E_0 - E_1 + E_L) * (E_L + E_R) * (E_2 - E_1 + E_R)) + 2 / ((E_2 - E_1 + E_L) * (E_L + E_R) * (E_0 - E_1 + E_R))
display(sympy.Eq(sympy.Symbol('I_{c}(n=1)'), I_1))

I_2 = 4 / ((E_1 - E_2 + E_L) * (E_0 - E_2) * (E_1 - E_2 + E_R)) + 2 / ((E_1 - E_2 + E_L) * (E_L + E_R) * (E_1 - E_2 + E_R))
display(sympy.Eq(sympy.Symbol('I_{c}(n=2)'), I_2))

# %%
I_1.subs(n_C, n_C + 1).expand().factor()

# %%
shifted = simplified_result.subs(n_C, n_C + 1)

# %%
shifted

# %%
I_1.expand().factor()

# %%
simplified_result

# %%
