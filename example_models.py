import sympy as sp
import numpy as np
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum import Dagger
from second_quantization import fock_space


def build_spinful_chain_hamiltonian(n):
    t, U, mu = sp.symbols("t U mu", real=True)
    ops_up = [FermionOp("c_{" + f"{i}" + "\\uparrow}") for i in range(n)]
    ops_dn = [FermionOp("c_{" + f"{i}" + "\\downarrow}") for i in range(n)]
    opers_sorted = []
    for idx in range(len(ops_up)):
        opers_sorted.append(ops_dn[idx])
        opers_sorted.append(ops_up[idx])

    hopping = 0
    interaction = 0
    chemical = 0

    for i in range(n - 1):
        # Hopping for up
        hopping += -t * (
            Dagger(ops_up[i]) * ops_up[i + 1] + Dagger(ops_up[i + 1]) * ops_up[i]
        )
        # Hopping for down
        hopping += -t * (
            Dagger(ops_dn[i]) * ops_dn[i + 1] + Dagger(ops_dn[i + 1]) * ops_dn[i]
        )

    for i in range(n):
        # Onsite Coulomb interaction
        interaction += (
            U * (Dagger(ops_up[i]) * ops_up[i]) * (Dagger(ops_dn[i]) * ops_dn[i])
        )
        # Chemical potential for both spins
        chemical += -mu * (
            Dagger(ops_up[i]) * ops_up[i] + Dagger(ops_dn[i]) * ops_dn[i]
        )

    H = hopping + interaction + chemical
    return H, ops_up + ops_dn, (t, U, mu)


def kitaev_chain(length):
    """
    Create a 1D fermionic chain Hamiltonian with nearest neighbor hopping, pairing
    and on-site interaction.
    Parameters:
    ----------
    length : int
        Length of the chain.
    Returns:
    -------
    tuple
        A tuple containing:
            - symbols: tuple of symbols (t, Delta, U, mus)
            - H: Hamiltonian operator in second quantization
            - names: list of fermionic operator names
    """

    names = ["c_{i}".format(i=i) for i in range(length)]
    fermions = [FermionOp(name) for name in names]

    t, Delta, U = sp.symbols("t Delta, U", real=True, positive=True)
    mu = sp.Symbol("mu", real=True)
    H = t * sum(Dagger(a) * b for a, b in zip(fermions, fermions[1:]))
    H += Delta * sum(Dagger(a) * Dagger(b) for a, b in zip(fermions, fermions[1:]))
    H += Dagger(H)

    for i in range(len(fermions)):
        H += mu * Dagger(fermions[i]) * fermions[i]

    # H += U * sum(Dagger(a) * a * Dagger(b) * b for a, b in zip(fermions, fermions[1:]))

    return (
        H,
        fermions,
        (t, Delta, mu),
    )


c_down = fock_space.FermionSequence([0])
c_up = fock_space.FermionSequence([2])
a_up = ~c_up
a_down = ~c_down

n_up = c_up * a_up
n_down = c_down * a_down
P = -c_down * a_down * c_up * a_up + 1  # Projects out double ocucpance

omega = sp.nsimplify(sp.exp(2 * sp.pi * sp.I / 3))

## Initial U and V matrices
U = sp.Matrix(3, 3, lambda i, j: 0)
U[0, 1] = 1
U[1, 2] = 1
U[2, 0] = 1

V = sp.Matrix(3, 3, lambda i, j: 0)
V[0, 0] = 1
V[1, 1] = omega
V[2, 2] = omega**2

base_vec = [
    [P * (-n_up - n_down + 1) * P, P * a_down * P, P * a_up * P],
    [P * c_down * P, P * n_down * P, P * c_down * a_up * P],
    [P * c_up * P, P * c_up * a_down * P, P * n_up * P],
]


def matrix_to_operator(matrix):
    oper = fock_space.FermionSequence()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            oper += matrix[i, j] * base_vec[i][j]
    oper.remove_zero_weight()
    return oper


def make_rotated_clock_operators(W, simplify=False):
    if simplify:
        U_prime = sp.simplify(W.H * U * W)
        V_prime = sp.simplify(W.H * V * W)
    else:
        U_prime = W.H * U * W
        V_prime = W.H * V * W
    sigma = matrix_to_operator(U_prime)
    tau = matrix_to_operator(V_prime)
    if simplify:
        sigma.simplify()
        tau.simplify()

    return sigma, tau


def split_odd_even(oper):
    oper_list_even = []
    oper_list_odd = []
    weight_list_even = []
    weight_list_odd = []
    for idx, w in enumerate(oper.oper_list):
        if len(w) % 2:
            oper_list_odd.append(w)
            weight_list_odd.append(oper.weights[idx])
        else:
            oper_list_even.append(w)
            weight_list_even.append(oper.weights[idx])

    even_op = fock_space.FermionSequence(
        oper_list_even, weights=weight_list_even, bypass_parse=True
    )
    odd_op = fock_space.FermionSequence(
        oper_list_odd, weights=weight_list_odd, bypass_parse=True
    )
    return even_op, odd_op


def shift_clock(clock_operator, N):
    JW_projector_base = (-2 * P * c_up * P * P * a_up * P + 1) * (
        -2 * P * c_down * P * a_down * P + 1
    )
    JW_projector = fock_space.FermionSequence(1)
    for i in range(N):
        JW_projector *= JW_projector_base >> i

    even_part, odd_part = split_odd_even(clock_operator)
    operator_graded = (even_part >> N) + JW_projector * (odd_part >> N)
    return operator_graded


alpha, beta, gamma, theta = sp.symbols("alpha beta gamma theta", real=True)
omega = sp.nsimplify(sp.exp(2 * sp.pi * sp.I / 3))


def general_W_for_sigma_even():
    W0 = sp.Matrix(
        [
            [1 / sp.sqrt(3), 1 / sp.sqrt(2), 1 / sp.sqrt(6)],
            [1 / sp.sqrt(3), -1 / sp.sqrt(2), 1 / sp.sqrt(6)],
            [1 / sp.sqrt(3), 0, -2 / sp.sqrt(6)],
        ]
    )
    U2 = sp.Matrix(
        [
            [sp.exp(sp.I * beta) * sp.cos(theta), sp.exp(sp.I * gamma) * sp.sin(theta)],
            [
                -sp.exp(-sp.I * gamma) * sp.sin(theta),
                sp.exp(-sp.I * beta) * sp.cos(theta),
            ],
        ]
    )
    U = sp.Matrix(
        [[sp.exp(sp.I * alpha), 0, 0], [0, U2[0, 0], U2[0, 1]], [0, U2[1, 0], U2[1, 1]]]
    )
    return W0 @ U


def clock_hamiltonian(N=2, alpha=0, f=0, J=1):
    phi = sp.symbols("phi", real=True)
    phase_factor = sp.exp(-sp.I * phi)
    W = general_W_for_sigma_even()
    W = W.subs(beta, 0).subs(gamma, 0).subs(alpha, 0).subs(theta, 0)
    sigma, tau = make_rotated_clock_operators(W)

    H = fock_space.FermionSequence()

    for i in range(N - 1):
        H += -J * (
            phase_factor * (~shift_clock(sigma, i)) * shift_clock(sigma, i + 1)
            + np.conj(phase_factor)
            * (~shift_clock(sigma, i + 1) * shift_clock(sigma, i))
        )
    for i in range(N):
        H += -f * (
            (~shift_clock(tau, i)) * phase_factor
            + shift_clock(tau, i) * np.conj(phase_factor)
        )
    H.remove_zero_weight()
    H.simplify()
    H_sympy, ops = H.to_sympy()
    return H_sympy, ops, phi
