#!/usr/bin/env python

import os
import sys

from setuptools import setup

import pypulse

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()


# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md')) as f:
    long_description = f.read()


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
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy', 'matplotlib', 'astropy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
