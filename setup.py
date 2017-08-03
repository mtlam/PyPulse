#!/usr/bin/env python

import os
import sys

from setuptools import setup

import pypulse

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


setup(
    name='PyPulse',
    version=pypulse.__version__,
    author='Michael Lam',
    author_email='michael.lam@nanograv.org',
    url='https://github.com/mtlam/PyPulse',
    packages=['pypulse'],
    package_dir = {'pypulse': 'pypulse'},
    zip_safe=False,
    license='GPLv3',
    description='A python package for handling and analyzing PSRFITS files.',
    install_requires=['numpy', 'scipy', 'astropy'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
