from setuptools import setup

setup(
	name='qiit',
	version=0.1,
	install_requires=[
	    'numpy',
        'qutip'
    ],
	author='Eric J. Michaud',
	author_email="ericjm@mit.edu",
	license='MIT',
	url='https://github.com/ejmichaud/quantum-iit',
	packages=["qiit"]
)

