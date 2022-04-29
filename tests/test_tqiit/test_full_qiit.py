
import math
from re import A

import numpy as np
import qutip as qt
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate

from qiit.tqiit import *
import pytest


def test_partitioned_channel1():
    U = qt.to_super(qt.identity([2, 2]))
    U_P = partitioned_channel(U, [0], [1], [0, 1])
    assert U_P.istp and U_P.iscp
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    mixed_dm = qt.maximally_mixed_dm(2)
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(zero_dm, one_dm))) == tprod(zero_dm, one_dm)
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(one_dm, zero_dm))) == tprod(one_dm, zero_dm)
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(one_dm, mixed_dm))) == tprod(one_dm, mixed_dm)

def test_partitioned_channel2():
    U = qt.to_super(qt.identity([2, 2, 2]))
    U_P = partitioned_channel(U, [0, 1], [2], [0, 1, 2])
    assert U_P.istp and U_P.iscp
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    mixed_dm = qt.maximally_mixed_dm(2)
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(zero_dm, mixed_dm, mixed_dm))) == tprod(zero_dm, mixed_dm, mixed_dm)
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(mixed_dm, zero_dm, one_dm))) == tprod(mixed_dm, zero_dm, one_dm)

def test_partitioned_channel3():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qt.to_super(qc.propagators(expand=True)[0])
    U_P = partitioned_channel(U, [0], [1], [0, 1])
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    mixed_dm = qt.maximally_mixed_dm(2)
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(zero_dm, one_dm))) == tprod(mixed_dm, mixed_dm) 
    assert qt.vector_to_operator(U_P * qt.operator_to_vector(tprod(one_dm, zero_dm))) == tprod(mixed_dm, mixed_dm)     

def test_II1():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, one_dm)
    assert II(qt.to_super(U), X, [0, 1]) == 0.5

def test_II2():
    U = qt.identity([2, 2])
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, one_dm)
    assert II(qt.to_super(U), X, [0, 1]) == 0.0

def test_II3():
    print("TODO: You deleted a test that was failing! Figure this out!")
    # zerozero = qt.basis([2, 2], [0, 0])
    # oneone = qt.basis([2, 2], [1, 1])
    # pure = (zerozero + oneone) / np.sqrt(2)
    # pure_dm = pure * pure.dag()
    # U = qt.identity([2, 2])
    # assert II(qt.to_super(U), pure_dm, [0, 1]) == 0.75

# def test_II4():
#     basis = [qt.basis(2, 0), qt.basis(2, 1)]
#     z, o = basis
#     n = 3
#     permutation = [0, 2, 1]
#     columns = []
#     phi_states = [z, o, z]
#     X = tprod(z, o, z)
#     for 
