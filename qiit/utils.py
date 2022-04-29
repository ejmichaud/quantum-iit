
from itertools import product

import numpy as np
import qutip as qt

# -------------------------------------- #
#           UTILITY FUNCTIONS
# -------------------------------------- #

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

def subsets(indices):
    """Generator of all subsets of a list of indices.
    """
    n = len(indices)
    for i in range(2**n):
        mask = [(i // (2**k)) % 2 for k in range(n)]
        subset = [indices[k] for k in range(n) if mask[k]]
        yield subset

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




