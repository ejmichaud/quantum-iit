
from itertools import product

import numpy as np
import qutip as qt
from qiit.utils import bipartitions, tprod, invert_permutation

def noise_factorized(rho, U, P1, P2, indices):
    """Applies to Psi the the operation U factorized along P1/P2 with noise.

    Args:
        rho: density matrix
        U: qutip superoperator
        P1 (list): half of partition
        P2 (list): other half of partition
        indices (list): indices of all parts of system (that we care about)
    
    Returns:
        density matrix
    """
    assert set(P1 + P2) == set(indices)
    assert U.issuper
    n = len(indices)
    n1 = len(P1)
    n2 = len(P2)
    noise_part = tprod(*[qt.maximally_mixed_dm(2) for _ in range(n - n1)])
    rho1 = tprod(qt.ptrace(rho, P1), noise_part).permute(invert_permutation(P1 + P2))
    Urho = qt.vector_to_operator(U * qt.operator_to_vector(rho1))
    U1rho = qt.ptrace(Urho, P1)
    noise_part = tprod(*[qt.maximally_mixed_dm(2) for _ in range(n - n2)])
    rho2 = tprod(qt.ptrace(rho, P2), noise_part).permute(invert_permutation(P2 + P1))
    Urho = qt.vector_to_operator(U * qt.operator_to_vector(rho2))
    U2rho = qt.ptrace(Urho, P2)
    return tprod(U1rho, U2rho).permute(invert_permutation(P1 + P2))

def myII(U, Psi, indices):
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
        UPsi = qt.vector_to_operator(U * qt.operator_to_vector(Psi))
        U_PPsi = noise_factorized(Psi, U, P1, P2, indices)
        rel_ent_dist = qt.entropy_relative(UPsi, U_PPsi, base=2)
        if rel_ent_dist < min_dist:
            min_dist = rel_ent_dist
    return min_dist



