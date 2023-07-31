from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='InteractionGroup',
    version='0.1',
    author='Sangmin Lee',
    author_email='smlee99@postech.ac.kr',
    license='MIT',
    long_description=read('README.md'),
    python_requires='>=3.6',
    install_requires=['numpy'],
    description='Energy decomposition of molecular dynamics simulation trajectories',
    packages=find_packages(),
)
