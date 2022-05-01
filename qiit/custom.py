
from itertools import product

import numpy as np
import qutip as qt
from qiit.utils import bipartitions, tprod, invert_permutation
from qiit.tqiit import partitioned_channel

# def noise_factorized(rho, U, P1, P2, indices):
#     """Applies to Psi the the operation U factorized along P1/P2 with noise.

#     Args:
#         rho: density matrix
#         U: qutip superoperator
#         P1 (list): half of partition
#         P2 (list): other half of partition
#         indices (list): indices of all parts of system (that we care about)
    
#     Returns:
#         density matrix
#     """
#     assert set(P1 + P2) == set(indices)
#     assert U.issuper
#     n = len(indices)
#     n1 = len(P1)
#     n2 = len(P2)
#     noise_part = tprod(*[qt.maximally_mixed_dm(2) for _ in range(n - n1)])
#     rho1 = tprod(qt.ptrace(rho, P1), noise_part).permute(invert_permutation(P1 + P2))
#     Urho = qt.vector_to_operator(U * qt.operator_to_vector(rho1))
#     U1rho = qt.ptrace(Urho, P1)
#     noise_part = tprod(*[qt.maximally_mixed_dm(2) for _ in range(n - n2)])
#     rho2 = tprod(qt.ptrace(rho, P2), noise_part).permute(invert_permutation(P2 + P1))
#     Urho = qt.vector_to_operator(U * qt.operator_to_vector(rho2))
#     U2rho = qt.ptrace(Urho, P2)
#     return tprod(U1rho, U2rho).permute(invert_permutation(P1 + P2))

def sjdiv(rho1, rho2, base=2):
    return qt.entropy_vn(0.5 * rho1 + 0.5 * rho2, base=base) \
                - 0.5 * qt.entropy_vn(rho1, base=base) - 0.5 * qt.entropy_vn(rho2, base=base)

def custom_sjdiv(rho1, rho2, base=2):
    M = 0.5 * (rho1 + rho2)
    return 0.5 * (qt.entropy_relative(rho1, M, base=base) + qt.entropy_relative(rho2, M, base=base))

def qIIns(U, Psi, indices, base=2):
    """A version of quantum integrated information.
    
    Minimizes Jensen-Shannon divergence between E(Psi) and E_P(Psi) over partitions P.

    Args:
        U: qutip superoperator
        Psi: density matrix
        indices (list): indices of all parts of the system (that we care about)
    
    Returns:
        (float)
    """
    assert U.issuper
    min_dist = float('inf')
    for P1, P2 in bipartitions(indices):
        # print(f"Partition: {P1}{P2}")
        UPsi = qt.vector_to_operator(U * qt.operator_to_vector(Psi))
        # print(f"U(Psi) = {UPsi}")
        U_P = partitioned_channel(U, P1, P2, indices)
        U_PPsi = qt.vector_to_operator(U_P * qt.operator_to_vector(Psi))
        # print(f"U_P(Psi) = {U_PPsi}")
        # U_PPsi = noise_factorized(Psi, U, P1, P2, indices)
        rel_ent_dist = sjdiv(UPsi, U_PPsi, base=base)
        if rel_ent_dist < min_dist:
            min_dist = rel_ent_dist
    return min_dist

def qIInk(U, Psi, indices, base=2):
    """A version of quantum integrated information.
    
    Minimizes relative entropy between E(Psi) and E_P(Psi) over partitions P.

    Args:
        U: qutip superoperator
        Psi: density matrix
        indices (list): indices of all parts of the system (that we care about)
    
    Returns:
        (float)
    """
    assert U.issuper
    min_dist = float('inf')
    for P1, P2 in bipartitions(indices):
        # print(f"Partition: {P1}{P2}")
        UPsi = qt.vector_to_operator(U * qt.operator_to_vector(Psi))
        # print(f"U(Psi) = {UPsi}")
        U_P = partitioned_channel(U, P1, P2, indices)
        U_PPsi = qt.vector_to_operator(U_P * qt.operator_to_vector(Psi))
        # print(f"U_P(Psi) = {U_PPsi}")
        # U_PPsi = noise_factorized(Psi, U, P1, P2, indices)
        rel_ent_dist = qt.entropy_relative(UPsi, U_PPsi, base=base)
        if rel_ent_dist < min_dist:
            min_dist = rel_ent_dist
    return min_dist

def qIInt(U, Psi, indices):
    """A version of quantum integrated information.
    
    Minimizes the trace distance between E(Psi) and E_P(Psi) over partitions P.

    Args:
        U: qutip superoperator
        Psi: density matrix
        indices (list): indices of all parts of the system (that we care about)
    
    Returns:
        (float)
    """
    assert U.issuper
    min_dist = float('inf')
    for P1, P2 in bipartitions(indices):
        # print(f"Partition: {P1}{P2}")
        UPsi = qt.vector_to_operator(U * qt.operator_to_vector(Psi))
        # print(f"U(Psi) = {UPsi}")
        U_P = partitioned_channel(U, P1, P2, indices)
        U_PPsi = qt.vector_to_operator(U_P * qt.operator_to_vector(Psi))
        # print(f"U_P(Psi) = {U_PPsi}")
        # U_PPsi = noise_factorized(Psi, U, P1, P2, indices)
        rel_ent_dist = qt.tracedist(UPsi, U_PPsi)
        if rel_ent_dist < min_dist:
            min_dist = rel_ent_dist
    return min_dist

def qIInb(U, Psi, indices):
    """A version of quantum integrated information.
    
    Minimizes the Bures between E(Psi) and E_P(Psi) over partitions P.

    Args:
        U: qutip superoperator
        Psi: density matrix
        indices (list): indices of all parts of the system (that we care about)
    
    Returns:
        (float)
    """
    assert U.issuper
    min_dist = float('inf')
    for P1, P2 in bipartitions(indices):
        # print(f"Partition: {P1}{P2}")
        UPsi = qt.vector_to_operator(U * qt.operator_to_vector(Psi))
        # print(f"U(Psi) = {UPsi}")
        U_P = partitioned_channel(U, P1, P2, indices)
        U_PPsi = qt.vector_to_operator(U_P * qt.operator_to_vector(Psi))
        # print(f"U_P(Psi) = {U_PPsi}")
        # U_PPsi = noise_factorized(Psi, U, P1, P2, indices)
        rel_ent_dist = qt.bures_dist(UPsi, U_PPsi)
        if rel_ent_dist < min_dist:
            min_dist = rel_ent_dist
    return min_dist


