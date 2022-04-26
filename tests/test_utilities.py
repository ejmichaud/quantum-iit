
import numpy as np
import qutip as qt
from qiit import *

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

