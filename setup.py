# -*- coding: utf-8 -*-
from setuptools import setup

__author__ = 'Martin Uhrin'
__license__ = 'GPLv3'

about = {}
with open('milad/version.py') as f:
    exec(f.read(), about)  # pylint: disable=exec-used

setup(
    name='milad',
    version=about['__version__'],
    description='Moment Invariants Local Atomic Descriptor',
    long_description=open('README.rst').read(),
    url='https://github.com/muhrin/milad.git',
    author='Martin Uhrin',
    author_email='martin.uhrin.10@ucl.ac.uk',
    license=__license__,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='machine learning, atomic descriptor, moment invariants',
    install_requires=[
        'dscribe', 'matplotlib', 'miniball', 'numba', 'numpy', 'rmsd', 'torch', 'scipy', 'sympy', 'scikit-learn',
        'scikit-image'
    ],
    extras_require={
        'dev': [
            'ipython',
            'pip',
            'pytest~=5.4',
            'pytest-cov',
            'pre-commit~=2.2',
            'prospector',
            'pylint==2.5.2',
            'twine',
            'yapf',
        ],
    },
    packages=['milad', 'milad.play'],
    include_package_data=True,
    test_suite='test',
    entry_points={
        'milad.plugins.types': ['milad_types = milad.provides:get_mince_types'],
    }
)
