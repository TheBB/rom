#!/usr/bin/env python

from setuptools import setup

setup(
    name='ROM',
    version='1.0.0',
    maintainer='Eivind Fonn',
    maintainer_email='evfonn@gmail.com',
    packages=['bbflow'],
    package_data={
        'bbflow': ['data/*.cpts'],
    },
    install_requires=[
        'click',
        'nutils',
        'numpy',
    ],
    entry_points={
        'console_scripts': ['bbflow=bbflow.__main__:main']
    },
)
