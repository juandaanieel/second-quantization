# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
os.chdir('/home/tinkerer/pmm_tools/codes/')
import numpy as np
import sympy as sy
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.operatorordering import normal_ordered_form

import matplotlib.pyplot as plt 
import hilbert_space as hs

from utils import *


# %%

def chain(length):

    names = [f'c_{{i}}'.format(i=i) for i in range(length)]
    fermions = [FermionOp(name) for name in names]

    t, Delta, U = sy.symbols('t Delta, U', real=True, positive=True)
    mus = sy.symbols(' '.join(f'mu_{i}' for i in range(length)), real=True)
    H = t * sum(Dagger(a) * b for a, b in zip(fermions, fermions[1:]))
    H += Delta * sum(Dagger(a) * Dagger(b) for a, b in zip(fermions, fermions[1:]))

    H += Dagger(H)
    
    for i, mu in enumerate(mus):
        H += mu * Dagger(fermions[i]) * fermions[i]

    H += U * sum(Dagger(a) * a * Dagger(b) * b for a, b in zip(fermions, fermions[1:]))

    
    return (t, Delta, U, mus), H, names



# %%

# %%
sizes = [2, 3, 4, 5]
spectra = []

for size in sizes:

    (t, Delta, U, mus), H, names = chain(size)
    fermions = [FermionOp(name) for name in names]

    _mus = np.linspace(-6, 3, 200)
    _Us = np.linspace(-1, 1, 201)
    _deltas = np.linspace(0, 3, 201)

    _t = 1
    _U = 3

    parameters = {t: _t, U: _U, Delta: _t}
    

    HS = hs.HilbertSpace()
    HS.add_fermion(names)
    HS.finalize()

    parity = HS.get_parity_op()
    terms = HS.matrix_numeric(H.expand())

    energies = []

    for _mu in _mus:
        aux = []
        for _delta in _deltas:

            params = parameters.copy()
            for mu in mus:
                params[mu] = _mu
            params[Delta] = _delta

            evals = np.sort(eigh_parity(params, parity, terms, return_eigenvectors=False)[0], axis=-1)
            gs_energy = evals[0, 0] - evals[1, 0]
            aux.append(gs_energy)

        energies.append(aux)        

    energies = np.array(energies)
    spectra.append(energies)
spectra = np.array(spectra)


# %%
fig, ax = plt.subplots(ncols=4, figsize=(10, 3))

for i, energies in enumerate(spectra):
    im = ax[i].pcolormesh(_deltas, _mus, energies, vmin=-1e-1, vmax=1e-1, cmap='PuOr')
    
    ax[i].set_xlabel(r'$\Delta/t$')

    if i > 0:
        ax[i].set_yticklabels([])
    else:
        ax[i].set_ylabel(r'$\mu/t$')

    ax[i].set_title(f'$L={sizes[i]}$')

    ax[i].scatter([_t + _U/2], [-_U/2],  c='red', s=2)

cbar = plt.colorbar(im , ax=ax, location='right', shrink=0.5)
cbar.set_label(r'$E_{\mathrm{gs}}$', rotation=0)


# %%
size = 3
sites = np.array(range(size))

(t, Delta, U, mus), H, names = chain(size)
fermions = [FermionOp(name) for name in names]

HS = hs.HilbertSpace(sparse=False)
HS.add_fermion(names)
HS.finalize()

parity = HS.get_parity_op()
terms = HS.matrix_numeric(H.expand())

parameters = {
    t: _t,
    U: _U,
    Delta: _t + _U/2,
}

for mu in mus:
    parameters[mu] = -_U/2

#parameters[mus[0]] = -_U/2 + .2
# parameters[mus[1]] = -_U
# parameters[mus[2]] = -_U

# %%
def partial_trace_generators(subset, complement):
    """
    Generate the projectors required to compute the partial trace of a subset of operators
    """
    subset_operators = HS.get_combinations(subset)
    complement_operators = HS.get_combinations(complement)

    size = list(subset_operators.values())[0].shape[0]
    vacuum = np.zeros(size)
    vacuum[0] = 1

    vecs = []

    for element in complement_operators.values():
        aux = []
        for g in subset_operators.values():
            generator = g @ element
            aux.append(vacuum @ generator)
        vecs.append(aux)

    return np.array(vecs)



# %%
evals, evecs = eigh_parity(parameters, parity, terms, return_eigenvectors=True)

even_gs = evecs[0, :, 0]
odd_gs = evecs[1, :, 0]

gamma_1 = np.outer(even_gs, odd_gs.conj()) + np.outer(odd_gs, even_gs.conj())
gamma_2 = 1j * (np.outer(even_gs, odd_gs.conj()) - np.outer(odd_gs, even_gs.conj()))

print(evals[0][0] - evals[1][0])
# coefficients_1, representation_1 = HS.get_operator_decomposition(gamma_1, get_basis=True)
# coefficients_2, representation_2 = HS.get_operator_decomposition(gamma_2, get_basis=True)
# majorana_1_symbolic = (sum(coef * op for coef, op in zip(coefficients_1, representation_1.keys()))).evalf(5)
# majorana_2_symbolic = sum(coef * op for coef, op in zip(coefficients_2, representation_2.keys())).evalf(5)


# %%
m1s = []
m2s = []

for i in range(len(fermions)):
    
    subset = [fermions[i]]
    complement = [ferm for ferm in fermions if ferm not in subset]
    PT = partial_trace_generators(subset, complement)

    reduced_gamma_1 = np.zeros((2, 2), dtype=complex)
    reduced_gamma_2 = np.zeros((2, 2), dtype=complex)

    for projector in PT:
        reduced_gamma_1 += projector @ gamma_1 @ projector.T.conj()
        reduced_gamma_2 += projector @ gamma_2 @ projector.T.conj()

    HS_reduced = hs.HilbertSpace()
    HS_reduced.add_fermion([names[i]])
    HS_reduced.finalize()

    reduced_majorana_1 = HS_reduced.get_operator_decomposition(reduced_gamma_1, get_basis=True)
    reduced_majorana_2 = HS_reduced.get_operator_decomposition(reduced_gamma_2, get_basis=True)
    try:
        m1 = float(reduced_majorana_1.simplify().args[0])
    except:
        m1 = 0
    try:
        m2 = float(reduced_majorana_2.simplify().args[0])
    except:
        m2 = 0

    m1s.append(m1)
    m2s.append(m2)

m1s = np.array(m1s)
m2s = np.array(m2s)


# %%
HS_reduced.get_operator_decomposition(reduced_gamma_1, get_basis=True)

# %%
fig, ax = plt.subplots(figsize=(5, 2))
ax.bar(sites - 0.1, np.abs(m1s), label=r'$\gamma_1$', alpha=1, width=0.1)
ax.bar(sites + 0.1, np.abs(m2s), label=r'$\gamma_2$', alpha=1, width=0.1)
ax.legend()
ax.set_xlabel('Site')
ax.set_ylabel('Majorana Coefficient')
ax.set_title(f'Chain of length $L$={size} with $U/t=${_U}')
ax.set_xticks(sites)


# %%
i = 0

subset = [fermions[i]]
complement = [ferm for ferm in fermions if ferm not in subset]
PT = partial_trace_generators(complement, subset)

reduced_gamma_1 = np.zeros((8, 8), dtype=complex)
reduced_gamma_2 = np.zeros((8, 8), dtype=complex)

for projector in PT:
    reduced_gamma_1 += projector @ gamma_1 @ projector.T.conj()
    reduced_gamma_2 += projector @ gamma_2 @ projector.T.conj()


# %%
HS_reduced = hs.HilbertSpace()
HS_reduced.add_fermion(names[1:])
HS_reduced.finalize()

reduced_majorana_1 = HS_reduced.get_operator_decomposition(reduced_gamma_1, get_basis=True)
reduced_majorana_2 = HS_reduced.get_operator_decomposition(reduced_gamma_2, get_basis=True)

reduced_majorana_1 = reduced_majorana_1.evalf(3).subs(Dagger(fermions[2]) * Dagger(fermions[1]) * fermions[2], - Dagger(fermions[1]) * Dagger(fermions[2]) * fermions[2])
reduced_majorana_1 = reduced_majorana_1.subs(Dagger(fermions[2]) * fermions[1] * fermions[2], - fermions[1] * Dagger(fermions[2]) * fermions[2])

reduced_majorana_2 = reduced_majorana_2.evalf(3).subs(Dagger(fermions[2]) * Dagger(fermions[1]) * fermions[2], - Dagger(fermions[1]) * Dagger(fermions[2]) * fermions[2])
reduced_majorana_2 = reduced_majorana_2.subs(Dagger(fermions[2]) * fermions[1] * fermions[2], - fermions[1] * Dagger(fermions[2]) * fermions[2])

# %%
reduced_majorana_1

# %%
reduced_majorana_2

# %%

# %%
