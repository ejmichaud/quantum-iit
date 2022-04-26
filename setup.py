from setuptools import setup

setup(
	name='qiit',
	version=0.1,
	install_requires=[
	    'numpy',
        'qutip'
    ],
	author='Eric J. Michaud',
	license='MIT',
	url='https://github.com/ejmichaud/quantum-iit',
	py_modules=['qiit']
)

