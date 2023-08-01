from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout, exit, stderr
import numpy as np

strpdb='../../pdbfiles/'
strff='../../forcefields/'
strdir='./'

temperature=300*kelvin
cutoff = 1.2*nanometer
freq = 1/picosecond
timestep = 1.0*femtoseconds


pdb = PDBFile('./init.pdb')

position = pdb.positions
boxvec = pdb.topology.getUnitCellDimensions().value_in_unit(nanometer)
zmax = boxvec[2]
numCells = 3
volume = boxvec[0]*boxvec[1]*boxvec[2]*numCells
pdb.topology.setUnitCellDimensions((boxvec[0],boxvec[1],zmax*numCells)*nanometer)
constEfield = (constV*elementary_charge*AVOGADRO_CONSTANT_NA/(zmax*nanometer)).in_units_of(kilojoule_per_mole/nanometer)

newposition = []*nanometer
for i in range(pdb.topology.getNumAtoms()):
    pos = position[i].value_in_unit(nanometer)
    newposition.append((pos[0],pos[1],pos[2])*nanometer)

newChain = pdb.topology.addChain()

for i,rs in enumerate(pdb.topology.residues()):
    if (rs.name == 'grp') and (rs.index < 641):
        for at in rs._atoms:
            newResidue = pdb.topology.addResidue('grp', newChain)
            pdb.topology.addAtom('CG',element.carbon,newResidue)
            pos = position[at.index].value_in_unit(nanometer)
            newposition.append((pos[0],pos[1],pos[2]+zmax)*nanometer)

position = newposition
PDBFile.writeFile(pdb.topology, position, open('twosurf.pdb', 'w'))

for i,rs in enumerate(pdb.topology.residues()):
    if rs.name == 'grp':
        for at in rs._atoms:
            listGRP.append(at.index)

forcefield = ForceField(strff+'charmm36.xml', strff+'charmm36_tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=cutoff, constraints=HBonds, rigidWater=True)

# Set force group
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]

nbondedForce.setNonbondedMethod(NonbondedForce.PME)
nbondedForce.setCutoffDistance(cutoff)
customNonbondedForce.setNonbondedMethod(customNonbondedForce.CutoffPeriodic)
print('nbMethod : ', customNonbondedForce.getNonbondedMethod())

nRealAtoms = system.getNumParticles()
boxvec = pdb.topology.getUnitCellDimensions()
volume = boxvec[0]*boxvec[1]*boxvec[2]

imsig = 1*nanometer
imeps = 0*kilojoule/mole
position = position.in_units_of(nanometer)

if constEfield._value != 0:
    constVforce = CustomExternalForce('efield*q*z')
    constVforce.addGlobalParameter('efield',constEfield._value)
    constVforce.addPerParticleParameter('q')
    for i in range(nRealAtoms):
        (q, sig, eps) = nbondedForce.getParticleParameters(i)
        constVforce.addParticle(i, [q])
    system.addForce(constVforce)

# slab dipole correction
cvdipole = CustomExternalForce('q*z')
cvdipole.addPerParticleParameter('q')

for i in range(system.getNumParticles()):
    (q, sig, eps) = nbondedForce.getParticleParameters(i)
    cvdipole.addParticle(i, [q])

cvforce = CustomCVForce("twopioverV*((muz)^2)")
cvforce.addCollectiveVariable("muz",cvdipole)
ONE_4PI_EPS0=138.935456
cvforce.addGlobalParameter("twopioverV",ONE_4PI_EPS0*2*np.pi/volume._value)
system.addForce(cvforce)

for i in range(len(listGRP)):
    for j in range(i+1,len(listGRP)):
        nbondedForce.addException(listGRP[i],listGRP[j],0*elementary_charge**2,imsig,imeps) # original grp plane
        customNonbondedForce.addExclusion(listGRP[i],listGRP[j])

for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)

platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}
properties["DeviceIndex"]="0"

# Initial simulation to put the system in window
print('Beginning NVT ensemble calculation...')

# Integrator
integ_md_1 = LangevinIntegrator(temperature, freq, timestep)

# NVT Simulation setup
simmd = Simulation(pdb.topology, system, integ_md_1, platform, properties)
simmd.context.setTime(0)
simmd.context.setPositions(position)
boxvec = simmd.topology.getPeriodicBoxVectors()
simmd.context.setPeriodicBoxVectors(boxvec[0],boxvec[1],boxvec[2])

temp_state = simmd.context.getState(getPositions=True,enforcePeriodicBox=True)
temp_position = temp_state.getPositions()
PDBFile.writeFile(simmd.topology, temp_position, open('md_nvt_init.pdb', 'w'))
print('md_nvt_init.pdb file is saved')

# Short NVT equilibration to adjust the biased potential
for i in range(1,initprodtime): #initprodtime
    simmd.step(1000)

simmd.context.setTime(0)
simmd.currentStep = 0
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True,getVelocities=True)
position = state.getPositions()
velocity = state.getVelocities()

del simmd.context

# Integrator
integ_md_2 = LangevinIntegrator(temperature, freq, timestep)

# NVT simulation setup
simmd = Simulation(pdb.topology, system, integ_md_2, platform, properties)
simmd.context.setPositions(position)
simmd.context.setVelocities(velocity)
boxvec = simmd.topology.getPeriodicBoxVectors()
simmd.context.setPeriodicBoxVectors(boxvec[0],boxvec[1],boxvec[2])

# Add reporters to simulation objects
simmd.reporters = []
simmd.reporters.append(DCDReporter(strdir+'md_nvt.dcd', 1000)) #1000
simmd.reporters.append(DCDReporter(strdir+'md_nvt_100.dcd', 50000))
simmd.reporters.append(StateDataReporter(strdir+'md_nvt.log', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, speed=True))
simmd.reporters.append(CheckpointReporter(strdir+'md_nvt.chk', 20000))
simmd.reporters[2].report(simmd,state)
enerlog=open(strdir+'md_nvt_ener.log', 'w')

# Write the header for the energy log file
enerlog.write('# Energy log file\n')
enerlog.write('# x1 : time (ps)\n')
for j in range(system.getNumForces()):
    f = system.getForce(j)
    enerlog.write('# x'+str(j+2) + ' : ' +str(type(f)) + ' (kJ/mol)\n')
print('Simulating window...')

print('Starting Production NVT Simulation...')
t1 = datetime.now()
for i in range(1,prodtime):
    print(i,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    simmd.step(1000)
    state = simmd.context.getState(getPositions=True)
    enerlog.write(str(i))
    for j in range(simmd.system.getNumForces()):
        f = simmd.system.getForce(j)
        enerlog.write('  ' + str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)))
    enerlog.write('\n')
    enerlog.flush()
enerlog.close()

t2 = datetime.now()
t3 = t2 - t1
print('Simulation took', t3.seconds,'seconds')

state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True,enforcePeriodicBox=True)
position = state.getPositions()
simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simmd.topology, position, open('md_nvt.pdb', 'w'))
print('Finished window...')
print('Simulation has finished!')

exit()
