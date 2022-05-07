<h1 align='center'> 
<i></i>
<code>qiit</code>

⚛️⚡🧠</h1>

Code for measuring *integration* in quantum systems. Includes several new measures and implements the measure from [Zandari et al. (2018)](https://arxiv.org/abs/1806.01421). 

## Installation
After cloning this repository, simply run:
```
python setup.py install
```
from the repository root directory and the `qiit` module will be installed. The only dependencies are numpy and qutip, which will be installed as part of the `qiit` installation if not already present.

## Usage

```python
import qutip as qt
from qutip.qip.algorithms import qft

from qiit.custom import qIInk

rho0 = qt.rand_dm(16, dims=[[2, 2, 2, 2], [2, 2, 2, 2]])
U = qft.qft(4)
print(qIInk(qt.to_super(U), rho0, [0, 1, 2, 3]))
```

## Testing
Several tests are included in the `tests` directory, which can be run with `pytest` from the repository root directory:
```
pytest
```

