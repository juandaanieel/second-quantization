import numpy as np
import numpy.linalg as nla
import itertools as it

def fermi_dist(E, mu, T):
    """
    Calculates the Fermi-Dirac distribution function.

    Parameters:
        E (float or numpy.ndarray): Energy.
        mu (float): Chemical potential of the leads.
        T (float): Temperature of the leads.

    Returns:
        float or numpy.ndarray: The value(s) of the Fermi-Dirac distribution function.
    """
    return 1.0 / (1.0 + np.exp((E - mu) / T))

def transition_matrix(evecs, c_m, lead_pos=[0, -1], spinful=True):
    """
    Calculates the transition matrices for electron tunneling.

    Parameters:
        evecs (numpy.ndarray): Eigenvectors of the system.
        c_m (list): List of fermionic operators representing the system.
        lead_pos (list, optional): Positions of the leads in the system. Defaults to [0, -1].
        spinful (bool, optional): Whether the system is spinful. Defaults to True.

    Returns:
        list: A list containing the transition matrices for electron tunneling.
              The list contains four elements: t1_dagger, t2_dagger, t1, and t2.
              - t1_dagger: Transition matrix for tunneling from lead 1 to the system.
              - t2_dagger: Transition matrix for tunneling from lead 2 to the system.
              - t1: Transition matrix for tunneling from the system to lead 1.
              - t2: Transition matrix for tunneling from the system to lead 2.
    """
    if spinful:
        t1_up_dagger = evecs.T.conj() @ c_m[lead_pos[0]].conj().T @ evecs
        t1_down_dagger = evecs.T.conj() @ c_m[lead_pos[0]+1].conj().T @ evecs
        t2_up_dagger = evecs.T.conj() @ c_m[lead_pos[1]-1].conj().T @ evecs
        t2_down_dagger = evecs.T.conj() @ c_m[lead_pos[1]].conj().T @ evecs

        t1_dagger = (
            np.asarray(np.abs(t1_up_dagger)) ** 2 + np.asarray(np.abs(t1_down_dagger)) ** 2
        )
        t2_dagger = (
            np.asarray(np.abs(t2_up_dagger)) ** 2 + np.asarray(np.abs(t2_down_dagger)) ** 2
        )
    else:
        t1_dagger = np.asarray(np.abs(evecs.T.conj() @ c_m[lead_pos[0]].conj().T @ evecs)**2)
        t2_dagger = np.asarray(np.abs(evecs.T.conj() @ c_m[lead_pos[1]].conj().T @ evecs)**2)

    t1 = t1_dagger.T
    t2 = t2_dagger.T

    return [t1_dagger, t2_dagger, t1, t2]

def gamma(Tlist, nlist, lead_params):
    """
    This function computes the tunneling rates (gamma) between the system and the leads based on the transmission probabilities and occupation probabilities of the leads.

    Parameters:
        Tlist (list): List containing the transmission probabilities for left and right leads.
        nlist (list): List containing the occupation probabilities for left and right leads.
        lead_params (dict): Dictionary containing lead parameters, including 'gammaL' (tunneling rate for the left lead) and 'gammaR' (tunneling rate for the right lead).

    Returns:
        tuple: A tuple containing the tunneling rates for the left and right leads.
               The tuple contains four elements: WL_dagger, WR_dagger, WL, and WR.
               - WL_dagger: Tunneling rate matrix for tunneling from the left lead to the system.
               - WR_dagger: Tunneling rate matrix for tunneling from the right lead to the system.
               - WL: Tunneling rate matrix for tunneling from the system to the left lead.
               - WR: Tunneling rate matrix for tunneling from the system to the right lead.

    Notes:
        - This function computes the tunneling rates by multiplying the transmission probabilities and occupation probabilities with the tunneling rates for the respective leads.
        - It ensures that the diagonal elements of the tunneling rate matrices are zero to avoid double counting.
    """    
    gammaL = lead_params["gammaL"]
    gammaR = lead_params["gammaR"]
    WL_dagger = gammaL * np.multiply(Tlist[0], nlist[0])
    WR_dagger = gammaR * np.multiply(Tlist[1], nlist[1])
    WL = gammaL * np.multiply(Tlist[2], nlist[2])
    WR = gammaR * np.multiply(Tlist[3], nlist[3])

    WL_dagger = WL_dagger - np.diag(np.diag(WL_dagger))
    WR_dagger = WR_dagger - np.diag(np.diag(WR_dagger))
    WL = WL - np.diag(np.diag(WL))
    WR = WR - np.diag(np.diag(WR))

    return WL_dagger, WR_dagger, WL, WR

def probability_vector(W_matrix):
    """
    This function calculates the probability vector representing the probabilities of occupying each state in the system in steady state, where the rate of change of probability is zero (dP/dt = 0).

    Parameters:
        W_matrix (numpy.ndarray): Transition rate matrix.

    Returns:
        numpy.ndarray: Probability vector in steady state.
    """
    N_state = W_matrix.shape[0]
    for i in range(N_state):
        W_matrix[i, i] = -np.sum(W_matrix[:, i])
    W_matrix = np.vstack((W_matrix, np.ones(N_state)))

    rhs = np.append(np.zeros(N_state), 1)
    P_vector = nla.pinv(W_matrix) @ rhs

    return P_vector

def current(Tlist, nlist, lead_params):
    """
    This function computes the currents flowing through the left and right leads based on the transmission probabilities, occupation probabilities, and lead parameters.

    Parameters:
        Tlist (list): List containing the transmission probabilities for left and right leads.
        nlist (list): List containing the occupation probabilities for left and right leads.
        lead_params (dict): Dictionary containing lead parameters, including 'gammaL' (tunneling rate for the left lead) and 'gammaR' (tunneling rate for the right lead).

    Returns:
        tuple: A tuple containing the currents flowing through the left and right leads, respectively.

    Notes:
        - This function first computes the tunneling rates using the gamma function based on the transmission probabilities, occupation probabilities, and lead parameters.
        - It then constructs the transition rate matrix (W_matrix) using the tunneling rates.
        - The probability vector in steady state (P_vector) is calculated using the probability_vector function.
        - Finally, the currents flowing through the left and right leads are computed based on the difference between the incoming and outgoing fluxes of electrons.
    """    
    WL_dagger, WR_dagger, WL, WR = gamma(Tlist, nlist, lead_params)
    W_matrix = WL_dagger + WL + WR_dagger + WR
    P_vector = probability_vector(W_matrix.astype(float)).T
    left_current = np.sum((WL_dagger.astype(float) - WL.astype(float)) @ P_vector)
    right_current = np.sum((WR_dagger.astype(float) - WR.astype(float)) @ P_vector)

    return left_current, right_current

def occupation_prob(evs, lead_params):
    """
    Calculates the occupation probabilities for a given many-body energy levels and lead parameters.

    Parameters:
        evs (numpy.ndarray): Eigenvalues of the system.
        lead_params (dict): Dictionary containing lead parameters, including 'VL' (left lead potential), 'VR' (right lead potential), and 'T' (temperature).

    Returns:
        list: A list containing the occupation probabilities for the left and right leads.
              The list contains four elements: nL_dagger, nR_dagger, nL, and nR.
              - nL_dagger: Occupation probability for states in the left lead above the Fermi level.
              - nR_dagger: Occupation probability for states in the right lead above the Fermi level.
              - nL: Occupation probability for states in the left lead below the Fermi level.
              - nR: Occupation probability for states in the right lead below the Fermi level.
    """
    VL, VR, T = lead_params["VL"], lead_params["VR"], lead_params["T"]
    energy_mesh = evs[:, None] - evs[None, :]
    
    nL_dagger = fermi_dist(energy_mesh, VL, T)
    nR_dagger = fermi_dist(energy_mesh, VR, T)
    nL = 1 - fermi_dist(-energy_mesh, VL, T)
    nR = 1 - fermi_dist(-energy_mesh, VR, T)

    return [nL_dagger, nR_dagger, nL, nR]

def zero_bias_conductance(ham, c_m, lead_params, spinful=True):
    """
    Calculates the zero-bias conductance matrix.

    Parameters:
        ham (numpy.ndarray): Many-body Hamiltonian of the system.
        c_dagger_m (tuple): Tuple of creation matrices.
        c_m (tuple): Tuple of annihilation matrices.
        lead_params (dict): Dictionary of lead parameters.
        spinful (bool, optional): Whether the system is spinful. Defaults to True.

    Returns:
        tuple: A tuple containing the four elements of the conductance matrix (G_LL, G_LR, G_RL, G_RR).
    """
    evs, evecs = nla.eigh(ham)
    Tlist = transition_matrix(evecs, c_m, spinful=spinful)
    _lead_params = {**lead_params}
    
    dV = _lead_params["dV"]
    _lead_params["VL"] = dV
    _lead_params["VR"] = 0.0
    nlist_L = occupation_prob(evs, _lead_params)
    I_left_V0, I_right_V0 = current(Tlist, nlist_L, _lead_params)

    _lead_params["VL"] = 0.0
    _lead_params["VR"] = dV
    nlist_R = occupation_prob(evs, _lead_params)
    I_left_0V, I_right_0V = current(Tlist, nlist_R, _lead_params)

    _lead_params["VL"] = 0.0
    _lead_params["VR"] = 0.0
    nlist_0 = occupation_prob(evs, _lead_params)
    I_left_00, I_right_00 = current(Tlist, nlist_0, _lead_params)

    G_LL = (I_left_V0 - I_left_00) / dV
    G_LR = (I_left_0V - I_left_00) / dV
    G_RL = (I_right_V0 - I_right_00) / dV
    G_RR = (I_right_0V - I_right_00) / dV

    return G_LL, G_LR, G_RL, G_RR

def finite_bias(ham, Vbias, c_m, lead_params, spinful=True):
    """
    Calculates the currents flowing through the system under finite bias conditions.

    Parameters:
        ham (numpy.ndarray): Many-body Hamiltonian of the system.
        Vbias (float): Voltage bias applied to the leads.
        c_dagger_m (tuple): Tuple of creation matrices.
        c_m (tuple): Tuple of annihilation matrices.
        lead_params (dict): Dictionary of lead parameters.
        spinful (bool, optional): Whether the system is spinful. Defaults to True.

    Returns:
        tuple: A tuple containing the four components of currents (I_left_V0, I_left_0V, I_right_V0, I_right_0V).
    """
    evs, evecs = nla.eigh(ham)
    Tlist = transition_matrix(evecs, c_m, spinful=spinful)

    _lead_params = {**lead_params}
    _lead_params["VL"] = Vbias
    _lead_params["VR"] = 0
    nlist_L = occupation_prob(evs, _lead_params)
    I_left_V0, I_right_V0 = current(Tlist, nlist_L, _lead_params)

    _lead_params["VL"] = 0
    _lead_params["VR"] = Vbias
    nlist_R = occupation_prob(evs, _lead_params)
    I_left_0V, I_right_0V = current(Tlist, nlist_R, _lead_params)

    return I_left_V0, I_left_0V, I_right_V0, I_right_0V

def get_conductances(ham, params_Bias, c_m, lead_params, detune_site=[0], spinful=True):
    """
    Compute the finite bias differential conductance matrix for the many-body Hamiltonian.

    This function calculates the finite bias differential conductance matrix for the many-body Hamiltonian when detuning the quantum dot on a specified site.

    Parameters:
        ham (numpy.ndarray): Many-body Hamiltonian of the system.
        params_Bias (dict): Dictionary containing bias parameters, including 'dmu' (array of chemical potential differences) and 'Vbias' (array of voltage biases).
        c_m (tuple): Tuple of creation and annihilation matrices.
        lead_params (dict): Dictionary of lead parameters.
        detune_site (list, optional): List of sites to detune the quantum dot. Defaults to [0].
        spinful (bool, optional): Whether the system is spinful. Defaults to True.

    Returns:
        numpy.ndarray: Finite bias differential conductance matrix in units of e^2/h.
    """
    
    dVbias = params_Bias["Vbias"][1] - params_Bias["Vbias"][0]
    
    currents = []
    
    for mu, Vbias in it.product(params_Bias["dmu"], params_Bias["Vbias"]):
        if spinful:
            ham_mu = ham + mu * sum([c_m[2*i].conj().T@c_m[2*i] + c_m[2*i+1].conj().T@c_m[2*i+1] for i in detune_site])
        else:
            ham_mu = ham + mu * sum([c_m[i].conj().T@c_m[i] for i in detune_site])
        currents.append(finite_bias(ham_mu, Vbias, c_m, lead_params, spinful=spinful))
    
    currents = np.array(currents).reshape(
        params_Bias["dmu"].shape[0], params_Bias["Vbias"].shape[0], 4
    )
    cond_matrix = 2*np.pi*np.gradient(currents, dVbias, axis=1)

    return cond_matrix
