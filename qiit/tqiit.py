
from itertools import product

import numpy as np
import qutip as qt
from qiit.utils import *


# -------------------------------------- #
#           SIMPLE IIT FUNCTIONS
# -------------------------------------- #

def noising_omega(X, omega, indices):
    """Noises the omega subsystem. Computes the partial trace over the `omega`
    parts and tensor products this with a maximally-mixed density matrix
    over the `omega` parts. This operation _forgets_ all information about
    the state of the `omega` subsystem.

    Args:
        X: density matrix
        omega (list): indices of parts of system to noise out
        indices (list): indices of all parts of the system
    Returns:
        A density matrix.
    """
    assert set(omega).issubset(set(indices))
    if not omega:
        return X
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(omega))])
    if set(omega) == set(indices):
        return noise_part
    Tr_Omega_X = qt.ptrace(X, complement(omega, indices))
    unordered_product = qt.tensor(Tr_Omega_X, noise_part)
    return unordered_product.permute(invert_permutation(complement(omega, indices) + omega))

def rho_e(P, M, U, Psi, indices):
    """Computes rho_e(P|M) for the given operation U and state Psi.

    If `P` is an empty set [], then returns a scalar, the complete trace of U(N_omega(Psi)).
    
    Args:
        P (list): indices of purview P
        M (list): indices of mechanism M
        U: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system
    Returns:
        A density matrix or scalar.
    """
    assert U.issuper
    N_Mp_Psi = noising_omega(Psi, complement(M, indices), indices)
    U_of_N = qt.vector_to_operator(U * qt.operator_to_vector(N_Mp_Psi))
    if not P:
        return np.trace(U_of_N)
    return qt.ptrace(U_of_N, P)

def rho_c(P, M, Ustar, Psi, indices):
    """Computes rho_c(P|M) for the given operation `Ustar` (adjoint of U) and state Psi.

    If `P` is an empty set [], then returns a scalar, the complete trace of U^*(N_omega(Psi)).
    
    Args:
        P (list): indices of purview P
        M (list): indices of mechanism M
        Ustar: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system
    Returns:
        A density matrix or scalar.
    """
    N_Mp_Psi = noising_omega(Psi, complement(M, indices), indices)
    Ustar_of_N = qt.vector_to_operator(Ustar * qt.operator_to_vector(N_Mp_Psi)) 
    if not P:
        return np.trace(Ustar_of_N)
    return qt.ptrace(Ustar_of_N, P)

def ei(P, M, U, Psi, indices):
    """Computes the effect information of M over P: ei(P|M).
    
    Args:
        P (list): indices of purview P
        M (list): indices of mechamism M
        U: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system
    Returns:
        (float): the trace distance between rho_e and maximally mixed state.
    """
    A = rho_e(P, M, U, Psi, indices)
    if not P:
        return qt.tracedist(qt.Qobj(A), qt.Qobj(1))
    B = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P))])
    return qt.tracedist(A, B)

def ci(P, M, Ustar, Psi, indices):
    """Computes the cause information of M over P: ci(P|M).
    
    Args:
        P (list): indices of purview P
        M (list): indices of mechanism M
        Ustar: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system
    Returns:
        (float): the trace distance between rho_c and maximally mixed state.
    """
    A = rho_c(P, M, Ustar, Psi, indices)
    if not P:
        return qt.tracedist(qt.Qobj(A), qt.Qobj(1))
    B = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P))])
    return qt.tracedist(A, B)

def phi_e(P, M, U, Psi, indices):
    """Computes "effect integrated information for mechanisms" phi^(e)(P|M).

    This is a minimum, over partitions of P and M (P1, P2), (M1, M2), of the 
    trace distance between rho_e(P|M) and the tensor product between 
    rho_e(P1|M1) and rho_e(P2|M2). Partitions where either (P1, M1) = ([], [])
    or (P1, M1) = (P, M) or (P2, M2) = ([], []) or (P2, M2) = (P, M) are not
    considered.

    Args:
        P (list): indices of purview P
        M (list): indices of mechanism M
        U: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system 
    Returns:
        (float)
    """
    best = float('inf')
    n = len(P) + len(M)
    if n == 0:
        return 0.0
    for i in range(1, 2**(n-1)):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        maskP = mask[:len(P)]
        maskM = mask[len(P):]
        P1 = [P[k] for k in range(len(P)) if maskP[k]]
        P2 = complement(P1, P)
        M1 = [M[k] for k in range(len(M)) if maskM[k]]
        M2 = complement(M1, M)
        rho_PM = rho_e(P, M, U, Psi, indices)
        rho_P1M1 = rho_e(P1, M1, U, Psi, indices)
        rho_P2M2 = rho_e(P2, M2, U, Psi, indices)
        rho_prod = tprod(rho_P1M1, rho_P2M2)
        if not P:
            dist = qt.tracedist(qt.Qobj(rho_PM), qt.Qobj(rho_prod))
        else:
            P1P2_indices = [P.index(x) for x in P1 + P2]
            rho_prod = rho_prod.permute(P1P2_indices)
            dist = qt.tracedist(rho_PM, rho_prod)
        if dist < best:
            best = dist
    return best if best != float('inf') else 0.0

def phi_c(P, M, Ustar, Psi, indices):
    """Computes "cause integrated information for mechanisms" phi^(c)(P|M).

    This is a minimum, over partitions of P and M (P1, P2), (M1, M2), of the 
    trace distance between rho_c(P|M) and the tensor product between 
    rho_e(P1|M1) and rho_e(P2|M2). Partitions where either (P1, M1) = ([], [])
    or (P1, M1) = (P, M) or (P2, M2) = ([], []) or (P2, M2) = (P, M) are not
    considered.

    Args:
        P (list): indices of purview P
        M (list): indices of mechanism M
        Ustar: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system 
    Returns:
        (float)
    """
    best = float('inf')
    n = len(P) + len(M)
    if n == 0:
        return 0.0
    for i in range(1, 2**(n-1)):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        maskP = mask[:len(P)]
        maskM = mask[len(P):]
        P1 = [P[k] for k in range(len(P)) if maskP[k]]
        P2 = complement(P1, P)
        M1 = [M[k] for k in range(len(M)) if maskM[k]]
        M2 = complement(M1, M)
        rho_PM = rho_c(P, M, Ustar, Psi, indices)
        rho_P1M1 = rho_c(P1, M1, Ustar, Psi, indices)
        rho_P2M2 = rho_c(P2, M2, Ustar, Psi, indices)
        rho_prod = tprod(rho_P1M1, rho_P2M2)
        if not P:
            dist = qt.tracedist(qt.Qobj(rho_PM), qt.Qobj(rho_prod))
        else:
            P1P2_indices = [P.index(x) for x in P1 + P2]
            rho_prod = rho_prod.permute(P1P2_indices)
            dist = qt.tracedist(rho_PM, rho_prod)
        if dist < best:
            best = dist
    return best if best != float('inf') else 0.0

def core_effect(M, U, Psi, indices):
    """Computes the purview P which maximizes phi^(e)(P|M) for the given
    mechanism M, quantum operation U, and state Psi.

    Args:    
        M (list): indices of mechanism M
        U: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system 
    Returns:
        tuple (P_star, max_phi) where P_star is the "core effect of M" and max_phi is the 
        effect integrated information it achieves.
    """
    max_phi = float('-inf')
    P_star = []
    n = len(indices)
    for i in range(2**n):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        P = [indices[k] for k in range(n) if mask[k]]
        phi_PM = phi_e(P, M, U, Psi, indices)
        if phi_PM > max_phi:
            max_phi = phi_PM
            P_star = P
    return P_star, max_phi

def core_cause(M, Ustar, Psi, indices):
    """Computes the purview P which maximizes phi^(c)(P|M) for the given
    mechanism M, quantum operation U, and state Psi.

    Args:    
        M (list): indices of mechanism M
        Ustar: superoperator representing trace-preserving unital quantum operation
        Psi: density matrix (system state)
        indices (list): indices for all parts of the system 
    Returns:
        tuple (P_star, max_phi) where P_star is the "core cause of M" and max_phi is the 
        cause integrated information it achieves.
    """
    max_phi = float('-inf')
    P_star = []
    n = len(indices)
    for i in range(2**n):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        P = [indices[k] for k in range(n) if mask[k]]
        phi_PM = phi_c(P, M, Ustar, Psi, indices)
        if phi_PM > max_phi:
            max_phi = phi_PM
            P_star = P
    return P_star, max_phi

def repertoire_e(M, U, Psi, indices):
    """Computes rho_e(P_star|M) tensor producted with the rest of the system
    maximally-mixed. here P_star is the core effect of M w.r.t. U and Psi.
    """
    P_star, integrated_e_phi = core_effect(M, U, Psi, indices)
    rho_e_star = rho_e(P_star, M, U, Psi, indices)
    P_starp = complement(P_star, indices)
    if len(P_starp) == 0:
        return rho_e_star, integrated_e_phi
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P_starp))])
    unordered_product = tprod(rho_e_star, noise_part)
    return unordered_product.permute(invert_permutation(P_star + P_starp)), integrated_e_phi

def repertoire_c(M, Ustar, Psi, indices):
    """Computes rho_e(P_star|M) tensor producted with the rest of the system
    maximally-mixed. here P_star is the core effect of M w.r.t. Ustar and Psi.
    """
    P_star, integrated_c_phi = core_cause(M, Ustar, Psi, indices)
    rho_c_star = rho_c(P_star, M, Ustar, Psi, indices)
    P_starp = complement(P_star, indices)
    if len(P_starp) == 0:
        return rho_c_star, integrated_c_phi
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P_starp))])
    unordered_product = tprod(rho_c_star, noise_part)
    return unordered_product.permute(invert_permutation(P_star + P_starp)),  integrated_c_phi

def integrated_phi(M, U, Ustar, Psi, indices):
    """Computes integrated cause/effect information of M.
    """
    _, integrated_e_phi = core_effect(M, U, Psi, indices)
    _, integrated_c_phi = core_cause(M, Ustar, Psi, indices)
    return min(integrated_e_phi, integrated_c_phi)

# -------------------------------------- #
#           FULL IIT FUNCTIONS
# -------------------------------------- #

def product_channel(U1, U2, P1, P2):
    """Computes product of two unital operations `U1`, `U2`.

    TODO: is there a permutation problem here?

    Args:
        U1: qutip superoperator
        U2: qutip superoperator
    Returns:
        qutip superoperator
    """
    assert U1.issuper and U2.issuper
    krauses1 = qt.to_kraus(U1)
    krauses2 = qt.to_kraus(U2)
    # Am I applying this permutation correctly?
    krauses_product = [qt.tensor(Vi, Vj).permute(invert_permutation(P1 + P2)) for Vi, Vj in product(krauses1, krauses2)]
    return qt.kraus_to_super(krauses_product)

def partitioned_channel(U, P1, P2, indices):
    """Partitions the channel U into a product channel.

    Args:
        U: qutip superoperator
        P1 (list): subset of `indices`
        P2 (list): subset of `indices`
        indices (list): indices of all parts of the system
    Returns:
        qutip superoperator
    """
    assert U.issuper
    assert set(P1 + P2) == set(indices)
    U1_columns = []
    n = len(indices)
    n1 = len(P1)
    n2 = len(P2)
    for i, j in product(range(2**n1), range(2**n1)):
        rho = np.zeros((2**n1, 2**n1))
        rho[j, i] = 1.0 # note that most of these aren't valid density matrices... I think this is fine...
        rho = qt.Qobj(rho, dims=[[2] * n1, [2] * n1])
        noise_part = tprod(*[qt.maximally_mixed_dm(2) for _ in range(n - n1)])
        rho = tprod(rho, noise_part).permute(invert_permutation(P1 + P2))
        Urho = qt.vector_to_operator(U * qt.operator_to_vector(rho))
        U1rho = qt.ptrace(Urho, P1)
        U1_columns.append(qt.operator_to_vector(U1rho))
    U2_columns = []
    for i, j in product(range(2**n2), range(2**n2)):
        rho = np.zeros((2**n2, 2**n2))
        rho[j, i] = 1.0
        rho = qt.Qobj(rho, dims=[[2] * n2, [2] * n2])
        noise_part = tprod(*[qt.maximally_mixed_dm(2) for _ in range(n - n2)])
        rho = tprod(rho, noise_part).permute(invert_permutation(P2 + P1))
        Urho = qt.vector_to_operator(U * qt.operator_to_vector(rho))
        U2rho = qt.ptrace(Urho, P2)
        U2_columns.append(qt.operator_to_vector(U2rho))
    U1 = qt.Qobj(np.array(U1_columns).reshape(4**n1, 4**n1), dims=[[[2] * n1, [2] * n1]] * 2)
    U2 = qt.Qobj(np.array(U2_columns).reshape(4**n2, 4**n2), dims=[[[2] * n2, [2] * n2]] * 2)
    return product_channel(U1, U2, P1, P2)
 
def CS(U, Ustar, Psi, indices):
    """Computes the conceptual structure CS(U).

    Args:
        U: qutip superoperator
        Ustar: qutip superoperator, the adjoint of the channel U
        Psi: quantum state
        indices (list): indices of all parts of the system
    Returns:
        dict tuple(M) -> (rho_c(M), rho_e(M), phi(M))
    """
    cs_dict = dict()
    for M in subsets(indices):
        rc, pc = repertoire_c(M, Ustar, Psi, indices)
        re, pe = repertoire_e(M, U, Psi, indices)
        if min(pe, pc) > 0:
            cs_dict[tuple(M)] = (rc, re, min(pe, pc))
    return cs_dict

def CS_distance(C1, C2, indices):
    total = 0
    for M in subsets(indices):
        if tuple(M) in C1 and tuple(M) in C2:
            rc1, re1, p1 = C1[tuple(M)]
            rc2, re2, p2 = C2[tuple(M)]
            total += (p1 * rc1 - p2 * rc2).norm()
            total += (p1 * re1 - p2 * re2).norm()
        if tuple(M) in C1 and tuple(M) not in C2:
            rc1, re1, p1 = C1[tuple(M)]
            total += (p1 * rc1).norm()
            total += (p1 * re1).norm()
        if tuple(M) not in C1 and tuple(M) in C2:
            rc2, re2, p2 = C2[tuple(M)]
            total += (p2 * rc2).norm()
            total += (p2 * re2).norm()
    return 0.25 * total

def II(U, Psi, indices):
    """The full global network integrated information.
    
    Args:
        U: qutip superoperator
        Psi: state
        indices (list): indices of all parts of the system
    
    Returns:
        (float): the integrated information
    """
    min_dist = float('inf')
    Ustar = superoperator_adjoint(U)
    for P1, P2 in bipartitions(indices):
        print(f"Partition: ({P1}, {P2})")
        U12 = partitioned_channel(U, P1, P2, indices)
        U12star = superoperator_adjoint(U12)
        C1 = CS(U, Ustar, Psi, indices)
        C2 = CS(U12, U12star, Psi, indices)
        print(f"Conceptual Structure 1: {C1}")
        print(f"Conceptual Structure 2: {C2}")
        d = CS_distance(C1, C2, indices)
        if d < min_dist:
            min_dist = d
    return min_dist
        

    # for (P1, P2), (M1, M2) in product(bipartitions(P), bipartitions(M)):
    #     print((P1, P2), (M1, M2))
    #     rho_PM = rho_e(P, M, U, Psi, indices)
    #     rho_P1M1 = rho_e(P1, M1, U, Psi, indices)
    #     rho_P2M2 = rho_e(P2, M2, U, Psi, indices)
    #     rho_prod = tprod(rho_P1M1, rho_P2M2)
    #     # this is a potential bug if I did this wrong
    #     rho_prod = rho_prod.permute(invert_permutation(P1 + P2))
    #     dist = qt.tracedist(rho_PM, rho_prod)
    #     print(dist)
    #     if dist < best:
    #         best = dist
    # return best


