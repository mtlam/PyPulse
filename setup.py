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
    download_url = 'https://github.com/mtlam/PyPulse/archive/v0.0.1.tar.gz',
    packages=['pypulse'],
    package_dir = {'pypulse': 'pypulse'},
    zip_safe=False,
    license='BSD-3',
    description='A python package for handling and analyzing PSRFITS files.',
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD-3 Clause License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
