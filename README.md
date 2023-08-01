## Introduction

InteractionGroup is a Python-based module for the energy decomposition of molecular dynamics simulation trajectories. Decomposing the nonbonded energy in various chemical systems rainging from interactions of drugs with proteins to novel materials is essential to the deeper understanding of molecular phenomena. 

Considering that the only total energy value is given in the energy logfile, accessing the interaction energy between specific group of atoms would be highly informative. This module read, and analyze MD trajectories in a post-process manner with only a few lines of Python code to do so.

> [!NOTE]\
> This module currently works only with a simulation package [openMM](https://github.com/openmm/openmm) version>=7.5.  

> [!WARNING]\
> There exist various types of force field with a plethora of nonbonded energy computation methods. Considering that this module is only based on the `Force` object created by `system` object rather than parsing force field xml file, one should always bear in mind that the energy defined by this module is in line with the force field file. One good way to check this is to compare the sum of decomposed nonbonded energy with the total nonbonded energy, as will be explained below.

## Information

PME/Cutoff ,Forcefield

## Example analysis script




## License

This project is licensed under the terms of the MIT license.
