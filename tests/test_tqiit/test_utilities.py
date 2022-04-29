
import numpy as np
import qutip as qt
from qiit.utils import *

import pytest

def test_complement1():
    assert complement([0, 1], [0, 1, 2, 3]) == [2, 3]
    assert complement([0, 2], [0, 1, 2]) == [1]
    assert complement([0, 3, 5], [0, 1, 2, 3, 4, 5]) == [1, 2, 4]
    assert complement([6, 7, 8], [6, 7, 8, 10]) == [10]

def test_invert_permutation1():
    assert invert_permutation([1, 0]) == [1, 0]
    assert invert_permutation([0, 1, 2, 3, 4]) == [0, 1, 2, 3, 4]
    assert invert_permutation([0, 2, 1, 3]) == [0, 2, 1, 3]
    assert invert_permutation([1, 2, 3, 0]) == [3, 0, 1, 2]

def test_bipartitions1():
    indices = [0, 1]
    assert len(list(bipartitions(indices))) == 1
    indices = [0, 1, 2]
    bps = list(bipartitions(indices))
    assert len(bps) == 3
    assert ([0], [1, 2]) in bps or ([1, 2], [0]) in bps
    assert ([1], [0, 2]) in bps or ([0, 2], [1]) in bps
    assert ([2], [0, 1]) in bps or ([0, 1], [2]) in bps

def test_subsets1():
    indices = [0, 1]
    ssets = list(subsets(indices))
    assert len(ssets) == 4
    assert [] in ssets
    assert [0] in ssets
    assert [1] in ssets
    assert [0, 1] in ssets

def test_tprod1():
    assert tprod(2.0, 3.0) == 6.0
    assert tprod(1.0, 0.0) == 0.0
    assert tprod(1.0, qt.maximally_mixed_dm(2)) == qt.maximally_mixed_dm(2)
    assert tprod(1.0, qt.maximally_mixed_dm(2), qt.maximally_mixed_dm(2)) == qt.tensor(qt.maximally_mixed_dm(2), qt.maximally_mixed_dm(2))
    dms = [qt.rand_dm(2) for _ in range(5)]
    assert tprod(*dms) == qt.tensor(*dms)

