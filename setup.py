import sys
from setuptools import setup, find_packages

setup_requires = []

install_requires = [
        "numpy",
        "attr",
        "scipy",
        "tinyfk>=0.4.4",
        "scikit-robot>=0.0.15",
        ]

setup(
    name='yamaopt',
    version='0.0.1',
    description='',
    license=license,
    install_requires=install_requires,
)
