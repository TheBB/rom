#!/usr/bin/env python

from setuptools import setup

setup(
    name='Aroma',
    version='1.1.0',
    maintainer='Eivind Fonn',
    maintainer_email='evfonn@gmail.com',
    packages=['aroma'],
    package_data={
        'aroma': ['data/*.cpts'],
    },
    install_requires=[
        'click',
        'h5py',
        'nutils>=4.0',
        'numpy',
        'scipy',
        'sharedmem',
    ],
)
