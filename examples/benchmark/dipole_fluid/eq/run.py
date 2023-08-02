from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from sys import stdout, exit, stderr
from simtk.unit import *
from simtk.openmm import app  # contains functions for i/o
import simtk.openmm as mm   # contains functions md work
from simtk import unit  # controls physical units
from sys import stdout  # print output to the terminal during simulation
from time import gmtime, strftime
from datetime import datetime
import threading
import os
import random
import numpy as np

# System info
strdir='./'
# Physical constant
temperature=300*kelvin
cutoff = 1.4*nanometer
freq = 1/picosecond
timestep = 1.0*femtoseconds
eqtime = int((1*nanosecond)/(1000*timestep)+.1)+1

pdb = PDBFile('../../../pdbfiles/box.pdb')

forcefield = ForceField('../../../forcefields/dipole.xml')

pdb.topology.loadBondDefinitions('../../../forcefields/dipole_residues.xml')
pdb.topology.createStandardBonds()

print(pdb.topology)

positions = pdb.positions
boxvec = pdb.topology.getUnitCellDimensions().value_in_unit(nanometer)

# Integrator
integ_eq = LangevinIntegrator(temperature, freq, timestep)

# System setup
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=cutoff, constraints=HBonds, ignoreExternalBonds=True, rigidWater=True)

# Set force group
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]

nbondedForce.setNonbondedMethod(NonbondedForce.PME)
nbondedForce.setCutoffDistance(cutoff)
customNonbondedForce.setNonbondedMethod(customNonbondedForce.CutoffPeriodic)
print('nbMethod : ', customNonbondedForce.getNonbondedMethod())

barostat = MonteCarloBarostat(1*atmosphere, temperature)
mcbarostat = system.addForce(barostat)

for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)

totmass = 0.*dalton
for i in range(system.getNumParticles()):
    totmass += system.getParticleMass(i)

# Create simulation objiect
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"] = "0";

simeq = Simulation(pdb.topology, system, integ_eq, platform, properties)
simeq.context.setPositions(pdb.positions)

# Read current state object to save the initial structure to 'beforemin.pdb' file
state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True)
positions = state.getPositions()
PDBFile.writeFile(simeq.topology, positions, open(strdir+'beforemin.pdb', 'w'))

# Minimize the energy
print('Wrote initial positions')
#simeq.minimizeEnergy(maxIterations=20000)

print('Minimization finished !')

# Shows how to print the energy terms and energy components (which is force group in OpenMM)
state = simeq.context.getState(getEnergy=True,getForces=True,getVelocities=True,getPositions=True)
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))

for i in range(system.getNumForces()):          # Each energy terms (harmonic bond, harmonic angle, etc...)
    f = system.getForce(i)
    print(type(f), str(simeq.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()))

# Save the minimized structure
positions = state.getPositions()
PDBFile.writeFile(simeq.topology, positions, open(strdir+'min.pdb', 'w'))

# First NPT equilibration 
print('Equilibrating...')
t1 = datetime.now()
simeq.context.setVelocitiesToTemperature(temperature)

# Set log files and log frequencies
dcdfile = strdir+'eq_npt.dcd'
logfile = strdir+'eq_npt.log'
chkfile = strdir+'eq_npt.chk'
templogfile = strdir+'eq_npt_temp.log'
simeq.reporters.append(DCDReporter(dcdfile, 1000))
simeq.reporters.append(StateDataReporter(logfile, 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, density=True,speed=True))
simeq.reporters.append(CheckpointReporter(chkfile, 50000))

# Report the first snapshot to dcd reporter and log file
simeq.reporters[0].report(simeq,state)
simeq.reporters[1].report(simeq,state)
print('Simulating...')

# Energy log file that reports energy components
enerlog = open('eq_npt_ener.log', 'w')
enerlog.write('# Energy log file\n')
enerlog.write('# x1 : time (ps)\n')

for j in range(system.getNumForces()):
    f = system.getForce(j)
    enerlog.write('# x'+str(j+2) + ' : ' +str(type(f)) + ' (kJ/mol)\n')

# Run NPT equilibration
for i in range(1,eqtime): #eqtime
    simeq.step(1000)
    print(i,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print(i,datetime.now())
    state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True, getVelocities=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))
    enerlog.write(str(i))
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        enerlog.write('  ' + str(simeq.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)))
    enerlog.write('\n')
    enerlog.flush()

enerlog.close()

t2 = datetime.now()
t3 = t2 - t1
print('equilibration is done !')
print('simulation took', t3.seconds,'seconds')

# Save final structure to 'eq_npt.pdb' file
state = simeq.context.getState(getEnergy=True,getForces=True,getPositions=True)
position = state.getPositions()

# Make sure that the new system size is properly updated to the simulation topology
simeq.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simeq.topology, position, open(strdir+'eq_npt.pdb', 'w'))
print('Wrote output coordinate file eq_npt.pdb')

# Report the final snapshot to checkpoint file
simeq.reporters[2].report(simeq,state)

# Remove the barostat
system.removeForce(mcbarostat)

exit()

