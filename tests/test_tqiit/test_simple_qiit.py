
import math

import numpy as np
import qutip as qt
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate

from qiit.tqiit import *
import pytest

def test_noising_omega1():
    X = tprod(*[qt.maximally_mixed_dm(2) for _ in range(5)])
    assert noising_omega(X, [0, 1, 2], [0, 1, 2, 3, 4]) == X
    assert noising_omega(X, [], [0, 1, 2, 3, 4]) == X
    assert noising_omega(X, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == X

def test_noising_omega2():
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, one_dm, qt.maximally_mixed_dm(2))
    assert noising_omega(X, [2], [0, 1, 2]) == X
    assert noising_omega(X, [0, 1], [0, 1, 2]) == tprod(*[qt.maximally_mixed_dm(2) for _ in range(3)])
    assert noising_omega(X, [0], [0, 1, 2]) == tprod(qt.maximally_mixed_dm(2), one_dm, qt.maximally_mixed_dm(2))

def test_rho_e():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    U_super = qt.to_super(U)
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, zero_dm)
    assert rho_e([0], [0], U_super, X, [0, 1]) == 0.5 * qt.identity(2)
    assert rho_e([1], [0], U_super, X, [0, 1]) == zero_dm
    assert rho_e([0, 1], [0], U_super, X, [0, 1]) == tprod(0.5 * qt.identity(2), zero_dm)
    assert rho_e([1], [1], U_super, X, [0, 1]) == 0.5 * qt.identity(2)
    assert rho_e([0], [1], U_super, X, [0, 1]) == zero_dm
    assert rho_e([0, 1], [1], U_super, X, [0, 1]) == tprod(zero_dm, 0.5 * qt.identity(2))
    assert rho_e([0], [0, 1], U_super, X, [0, 1]) == zero_dm
    assert rho_e([1], [0, 1], U_super, X, [0, 1]) == zero_dm
    assert rho_e([0, 1], [0, 1], U_super, X, [0, 1]) == tprod(zero_dm, zero_dm)

def test_rho_c():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    U_super = qt.to_super(U)
    U_super_star = superoperator_adjoint(U_super)
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, zero_dm)
    assert rho_c([0], [0], U_super_star, X, [0, 1]) == 0.5 * qt.identity(2)
    assert rho_c([1], [0], U_super_star, X, [0, 1]) == zero_dm
    assert rho_c([0, 1], [0], U_super_star, X, [0, 1]) == tprod(0.5 * qt.identity(2), zero_dm)
    assert rho_c([1], [1], U_super_star, X, [0, 1]) == 0.5 * qt.identity(2)
    assert rho_c([0], [1], U_super_star, X, [0, 1]) == zero_dm
    assert rho_c([0, 1], [1], U_super_star, X, [0, 1]) == tprod(zero_dm, 0.5 * qt.identity(2))
    assert rho_c([0], [0, 1], U_super_star, X, [0, 1]) == zero_dm
    assert rho_c([1], [0, 1], U_super_star, X, [0, 1]) == zero_dm
    assert rho_c([0, 1], [0, 1], U_super_star, X, [0, 1]) == tprod(zero_dm, zero_dm)

def test_ei1():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    U_super = qt.to_super(U)
    U_super_star = superoperator_adjoint(U_super)
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, zero_dm)
    assert ei([0], [0], U_super, X, [0, 1]) == 0.0
    assert ei([1], [1], U_super, X, [0, 1]) == 0.0
    assert ei([1], [0], U_super, X, [0, 1]) == 0.5
    assert ei([0], [1], U_super, X, [0, 1]) == 0.5
    assert ei([0], [0, 1], U_super, X, [0, 1]) == 0.5
    assert ei([1], [0, 1], U_super, X, [0, 1]) == 0.5
    assert ei([0, 1], [0, 1], U_super, X, [0, 1]) == 0.75
    
def test_ci1():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    U_super = qt.to_super(U)
    U_super_star = superoperator_adjoint(U_super)
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, zero_dm)
    assert math.isclose(ci([0], [0], U_super_star, X, [0, 1]), 0.0)
    assert math.isclose(ci([1], [1], U_super_star, X, [0, 1]), 0.0)
    assert math.isclose(ci([1], [0], U_super_star, X, [0, 1]), 0.5)
    assert math.isclose(ci([0], [1], U_super_star, X, [0, 1]), 0.5)
    assert math.isclose(ci([0], [0, 1], U_super_star, X, [0, 1]), 0.5)
    assert math.isclose(ci([1], [0, 1], U_super_star, X, [0, 1]), 0.5)
    assert math.isclose(ci([0, 1], [0, 1], U_super_star, X, [0, 1]), 0.75)

def test_phi_e1():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    U_super = qt.to_super(U)
    U_super_star = superoperator_adjoint(U_super)
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, zero_dm)
    
    assert phi_e([0], [0], U_super, X, [0, 1]) == 0.0
    assert phi_e([1], [1], U_super, X, [0, 1]) == 0.0
    
    assert rho_e([0], [0, 1], U_super, X, [0, 1]) == tprod(rho_e([0], [1], U_super, X, [0, 1]), rho_e([], [0], U_super, X, [0, 1]))
    assert rho_e([1], [0, 1], U_super, X, [0, 1]) == tprod(rho_e([1], [0], U_super, X, [0, 1]), rho_e([], [1], U_super, X, [0, 1]))
    assert rho_e([0, 1], [0], U_super, X, [0, 1]) == tprod(rho_e([0], [], U_super, X, [0, 1]), rho_e([1], [0], U_super, X, [0, 1]))
    assert rho_e([0, 1], [1], U_super, X, [0, 1]) == tprod(rho_e([0], [1], U_super, X, [0, 1]), rho_e([1], [], U_super, X, [0, 1]))
    assert rho_e([0, 1], [0, 1], U_super, X, [0, 1]) == tprod(rho_e([0], [1], U_super, X, [0, 1]), rho_e([1], [0], U_super, X, [0, 1]))

    assert phi_e([0, 1], [0], U_super, X, [0, 1]) == 0.0
    assert phi_e([0, 1], [1], U_super, X, [0, 1]) == 0.0
    assert phi_e([0], [0, 1], U_super, X, [0, 1]) == 0.0
    assert phi_e([1], [0, 1], U_super, X, [0, 1]) == 0.0
    assert phi_e([0, 1], [0, 1], U_super, X, [0, 1]) == 0.0

    assert phi_e([1], [0], U_super, X, [0, 1]) == 0.5
    assert phi_e([0], [1], U_super, X, [0, 1]) == 0.5

def test_phi_c1():
    qc = QubitCircuit(N=2, num_cbits=0)
    swap_gate = Gate(name="SWAP", targets=[0, 1])
    qc.add_gate(swap_gate)
    U = qc.propagators(expand=True)[0]
    U_super = qt.to_super(U)
    U_super_star = superoperator_adjoint(U_super)
    zero_dm = qt.basis(2, 0) * qt.basis(2, 0).dag()
    one_dm = qt.basis(2, 1) * qt.basis(2, 1).dag()
    X = tprod(zero_dm, zero_dm)
    
    assert phi_c([0], [0], U_super_star, X, [0, 1]) == 0.0
    assert phi_c([1], [1], U_super_star, X, [0, 1]) == 0.0
    
    assert rho_c([0], [0, 1], U_super_star, X, [0, 1]) == tprod(rho_c([0], [1], U_super_star, X, [0, 1]), rho_c([], [0], U_super_star, X, [0, 1]))
    assert rho_c([1], [0, 1], U_super_star, X, [0, 1]) == tprod(rho_c([1], [0], U_super_star, X, [0, 1]), rho_c([], [1], U_super_star, X, [0, 1]))
    assert rho_c([0, 1], [0], U_super_star, X, [0, 1]) == tprod(rho_c([0], [], U_super_star, X, [0, 1]), rho_c([1], [0], U_super_star, X, [0, 1]))
    assert rho_c([0, 1], [1], U_super_star, X, [0, 1]) == tprod(rho_c([0], [1], U_super_star, X, [0, 1]), rho_c([1], [], U_super_star, X, [0, 1]))
    assert rho_c([0, 1], [0, 1], U_super_star, X, [0, 1]) == tprod(rho_c([0], [1], U_super_star, X, [0, 1]), rho_c([1], [0], U_super_star, X, [0, 1]))

    assert phi_c([0, 1], [0], U_super_star, X, [0, 1]) == 0.0
    assert phi_c([0, 1], [1], U_super_star, X, [0, 1]) == 0.0
    assert phi_c([0], [0, 1], U_super_star, X, [0, 1]) == 0.0
    assert phi_c([1], [0, 1], U_super_star, X, [0, 1]) == 0.0
    assert phi_c([0, 1], [0, 1], U_super_star, X, [0, 1]) == 0.0

    assert phi_c([1], [0], U_super_star, X, [0, 1]) == 0.5
    assert phi_c([0], [1], U_super_star, X, [0, 1]) == 0.5


