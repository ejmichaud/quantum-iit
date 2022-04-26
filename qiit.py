
from itertools import product
from re import A

import numpy as np
import qutip as qt

# def trace(qobj):
#     return qt.Qobj(np.trace(qobj))

def tprod(*args):
    cs = [x for x in args if type(x) is not qt.Qobj]
    qs = [x for x in args if type(x) is qt.Qobj]
    constant_product = 1
    for c in cs:
        constant_product *= c
    if not qs:
        return constant_product
    return constant_product * qt.tensor(*qs)

def superoperator_adjoint(U):
    assert U.issuper
    krauses = qt.to_kraus(U)
    krauses_adjoint = [V.dag() for V in krauses]
    return qt.kraus_to_super(krauses_adjoint)

def product_channel(U1, U2):
    assert U1.issuper and U2.issuper
    krauses1 = qt.to_kraus(U1)
    krauses2 = qt.to_kraus(U2)
    krauses_product = [qt.tensor(Vi, Vj) for Vi, Vj in product(krauses1, krauses2)]
    return qt.kraus_to_super(krauses_product)

def partitioned_channel(U, P1, P2):


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
    return [x for x in indices if x not in subset]
        
def invert_permutation(perm):
    assert len(perm) == len(set(perm))
    result = [None] * len(perm)
    for i in range(len(perm)):
        result[perm[i]] = i
    return result

def noising_omega(X, omega, indices):
    """Noises the omega subsystem."""
    assert set(omega).issubset(set(indices))
    if not omega:
        return X
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(omega))])
    if set(omega) == set(indices):
        return noise_part
    Tr_Omega_X = qt.ptrace(X, complement(omega, indices))
    unordered_product = qt.tensor(Tr_Omega_X, noise_part)
    return unordered_product.permute(invert_permutation(complement(omega, indices) + omega))

# I assume that a partial trace over everything is basically just taking a complete trace, returning a float, not a Qobj
def rho_e(P, M, U, Psi, indices):
    assert U.issuper
    N_Mp_Psi = noising_omega(Psi, complement(M, indices), indices)
    N_Mp_Psi
    U_of_N = U * N_Mp_Psi * U.dag()
    if not P:
        return np.trace(U_of_N)
    return qt.ptrace(U_of_N, P)

def rho_c(P, M, U, Psi, indices):
    N_Mp_Psi = noising_omega(Psi, complement(M, indices), indices)
    U_of_N = U.dag() * N_Mp_Psi * U
    if not P:
        return np.trace(U_of_N)
    return qt.ptrace(U_of_N, P)

def ei(P, M, U, Psi, indices):
    A = rho_e(P, M, U, Psi, indices)
    if not P:
        return qt.tracedist(qt.Qobj(A), qt.Qobj(1))
    B = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P))])
    return qt.tracedist(A, B)

def ci(P, M, U, Psi, indices):
    A = rho_c(P, M, U, Psi, indices)
    if not P:
        return qt.tracedist(qt.Qobj(A), qt.Qobj(1))
    B = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P))])
    return qt.tracedist(A, B)

def phi_e(P, M, U, Psi, indices):
    best = float('inf')
    n = len(P) + len(M)
    for i in range(1, 2**(n-1)):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        maskP = mask[:len(P)]
        maskM = mask[len(P):]
        # print(mask)
        # print(maskP)
        # print(maskM)
        P1 = [P[k] for k in range(len(P)) if maskP[k]]
        P2 = complement(P1, P)
        M1 = [M[k] for k in range(len(M)) if maskM[k]]
        M2 = complement(M1, M)
        # print(f"phi_e computing partitions P={P} <> [P1][P2]={P1}{P1}, M={M} <> [M1][M2]={M1}{M2}")
        rho_PM = rho_e(P, M, U, Psi, indices)
        rho_P1M1 = rho_e(P1, M1, U, Psi, indices)
        rho_P2M2 = rho_e(P2, M2, U, Psi, indices)
        rho_prod = tprod(rho_P1M1, rho_P2M2)
        # this is a potential bug if I did this wrong
        # print(P1)
        # print(P2)
        # print(P1 + P2)
        # P_indices = [P.index(x) for x in P]
        if not P:
            dist = qt.tracedist(qt.Qobj(rho_PM), qt.Qobj(rho_prod))
        else:
            P1P2_indices = [P.index(x) for x in P1 + P2]
            rho_prod = rho_prod.permute(P1P2_indices)
            dist = qt.tracedist(rho_PM, rho_prod)
        # print(dist)
        if dist < best:
            best = dist
    return best if best != float('inf') else 0.0

def phi_c(P, M, U, Psi, indices):
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
        rho_PM = rho_c(P, M, U, Psi, indices)
        rho_P1M1 = rho_c(P1, M1, U, Psi, indices)
        rho_P2M2 = rho_c(P2, M2, U, Psi, indices)
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
    max_phi = float('-inf')
    P_star = []
    n = len(indices)
    for i in range(2**n):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        P = [indices[k] for k in range(n) if mask[k]]
        # print(f"core_effect sending P={P} into phi_e")
        phi_PM = phi_e(P, M, U, Psi, indices)
        # print(phi_PM)
        if phi_PM > max_phi:
            max_phi = phi_PM
            P_star = P
    return P_star, max_phi

def core_cause(M, U, Psi, indices):
    max_phi = float('-inf')
    P_star = []
    n = len(indices)
    for i in range(2**n):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        P = [indices[k] for k in range(n) if mask[k]]
        # print(P)
        phi_PM = phi_c(P, M, U, Psi, indices)
        # print(phi_PM)
        if phi_PM > max_phi:
            max_phi = phi_PM
            P_star = P
    return P_star, max_phi

def repertoire_e(M, U, Psi, indices):
    P_star, _ = core_effect(M, U, Psi, indices)
    rho_e_star = rho_e(P_star, M, U, Psi, indices)
    P_starp = complement(P_star, indices)
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P_starp))])
    unordered_product = tprod(rho_e_star, noise_part)
    return unordered_product.permute(invert_permutation(P_star + P_starp))

def repertoire_c(M, U, Psi, indices):
    P_star, _ = core_effect(M, U, Psi, indices)
    rho_e_star = rho_c(P_star, M, U, Psi, indices)
    P_starp = complement(P_star, indices)
    noise_part = qt.tensor(*[qt.states.maximally_mixed_dm(2) for _ in range(len(P_starp))])
    unordered_product = tprod(rho_e_star, noise_part)
    return unordered_product.permute(invert_permutation(P_star + P_starp))

def integrated_phi(M, U, Psi, indices):
    _, integrated_e_phi = core_effect(M, U, Psi, indices)
    _, integrated_c_phi = core_cause(M, U, Psi, indices)
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


