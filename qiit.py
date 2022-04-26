
from itertools import product
from re import A

import numpy as np
import qutip as qt

# def trace(qobj):
#     return qt.Qobj(np.trace(qobj))

def tprod(*args):
    """Computes tensor product of *args. The key behavior difference
    from `qt.tensor` is that it can handle scalars, performing 
    scalar multiplication instead of tensor multiplication when
    necessary.
    """
    cs = [x for x in args if type(x) is not qt.Qobj]
    qs = [x for x in args if type(x) is qt.Qobj]
    constant_product = 1
    for c in cs:
        constant_product *= c
    if not qs:
        return constant_product
    return constant_product * qt.tensor(*qs)

def superoperator_adjoint(U):
    """Computes adjoint of unital operation `U`.
    """
    assert U.issuper
    krauses = qt.to_kraus(U)
    krauses_adjoint = [V.dag() for V in krauses]
    return qt.kraus_to_super(krauses_adjoint)

def product_channel(U1, U2):
    """Computes product of two unital operations `U1`, `U2`.

    Args:
        U1: qutip superoperator
        U2: qutip superoperator
    Returns:
        qutip superoperator
    """
    assert U1.issuper and U2.issuper
    krauses1 = qt.to_kraus(U1)
    krauses2 = qt.to_kraus(U2)
    krauses_product = [qt.tensor(Vi, Vj) for Vi, Vj in product(krauses1, krauses2)]
    return qt.kraus_to_super(krauses_product)

# def partitioned_channel(U, P1, P2):


def bipartitions(indices):
    """Generator of bipartitions of a list of indices.
    
    Yields 2-tuples of lists, which are disjoint, and whose union is `indices`.
    Does not yield the ([], `indices`) bipartition. If len(indices) = n then 
    yields 2^(n-1) - 1 bipartitions.
    """
    n = len(indices)
    for i in range(1, 2**(n-1)):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        partition_A = [indices[j] for j in range(n) if mask[j]]
        partition_B = [indices[j] for j in range(n) if not mask[j]]
        yield (partition_A, partition_B)

def complement(subset, indices):
    """Returns complement of `subset` w.r.t. `indices`.
    """
    return [x for x in indices if x not in subset]
        
def invert_permutation(perm):
    """Inverts the permutation perm. `perm` specifies a permutation
    via a list, where i gets mapped to perm[i] for i in 0,...,n-1.
    """
    assert len(perm) == len(set(perm))
    result = [None] * len(perm)
    for i in range(len(perm)):
        result[perm[i]] = i
    return result

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
    P_star, _ = core_effect(M, U, Psi, indices)
    rho_e_star = rho_e(P_star, M, U, Psi, indices)
    P_starp = complement(P_star, indices)
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P_starp))])
    unordered_product = tprod(rho_e_star, noise_part)
    return unordered_product.permute(invert_permutation(P_star + P_starp))

def repertoire_c(M, Ustar, Psi, indices):
    """Computes rho_e(P_star|M) tensor producted with the rest of the system
    maximally-mixed. here P_star is the core effect of M w.r.t. Ustar and Psi.
    """
    P_star, _ = core_cause(M, Ustar, Psi, indices)
    rho_e_star = rho_c(P_star, M, Ustar, Psi, indices)
    P_starp = complement(P_star, indices)
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P_starp))])
    unordered_product = tprod(rho_e_star, noise_part)
    return unordered_product.permute(invert_permutation(P_star + P_starp))

def integrated_phi(M, U, Ustar, Psi, indices):
    """Computes integrated cause/effect information of M.
    """
    _, integrated_e_phi = core_effect(M, U, Psi, indices)
    _, integrated_c_phi = core_cause(M, Ustar, Psi, indices)
    return min(integrated_e_phi, integrated_c_phi)






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


