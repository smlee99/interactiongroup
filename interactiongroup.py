"""
InteractionGroup.py: Construct Interaction Group in system object based on an XML force field description

This is plugin for the OpenMM molecular simulation toolkit which is aimed
to decompose the nonbonded energy by analyzing the dcd trajectory in
postprocess manner. This module currently supports the nonbonded energy
for LennardJones, Electrostatic with NoCutoff and with Cutoff, Particle
Mesh Ewald - Direct Space interaction.

"""

from __future__ import absolute_import, print_function

__author__ = "Sangmin Lee"
__version__ = "1.0"

import os
import itertools
import math
import simtk.openmm as mm
import simtk.openmm.app as app
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, log, erfc
from itertools import product
from copy import deepcopy
from collections import defaultdict
import logging
import sys

#******************************* README  *************************************
#  This module defines methods that introduce the interaction energy between
#  two assigned groups. This module is aimed to decompose the total energy
#  which is defined by the force field parameter sets. Note that the current
#  settings are not containing proper exclusions for the 1-4 interaction, 
#  which implies that assessing the intra-molecular energy is not available. 
#  Electrostatic energies contain both Coulombic or PME - Direct Space term.
#*****************************************************************************

#***************************** WARGNINGS *************************************
#  Please note that there are various types of forcefield which define the
#  noncanonical analytic energy expression. Therefore, the usage of this code
#  should be taken with great care, as this code assumes the generic form
#  of Electrostatic and LennardJones energy expression. For the users who are
#  supposed to apply this code to decompose the nonbonded energy, please make
#  sure that this code is in line with the force field energy expression.
#*****************************************************************************

class InteractionGroup(object):
    """
    Interaction Group generator to construct the LennardJones / Electrostatic interaction
    between two assigned interaction group.
    """

    ONE_4PI_EPS0 = 138.935456

    def __init__(self, system, customnbforce='', useDispersionCorrection=True):
        """Load the system object and create general nonbonded force settings.

        Parameters
        ----------
        system : OpenMM system
            A system object created by the forcefield. Note that defining the InteractionGroup
            object should be done after all forces and exclusions/exceptions are made. General
            nonbonded force settings (i.e., cutoff distance) are then be extracted from the
            nonbondedForce and (if exist) customNonbondedForce in system object.
        option : customNonbondedForce type
            Option for the customNonbondedForce. This is generally regarded as the LennardJones
            force, including the NBFIX one. Note that there are various types of forcefield which
            also define the customNonbondedForce but with different analytic expression. This
            module currently supports only the LennardJonesForce.
        """
        
        self.DispersionCorrection = useDispersionCorrection
        self.customFlag = False
        self.customnbFlag = False
        self.groupForce = []
        self.groupForceName = []
        self.groupForceIndex = []
        
        listforce = [system.getForce(i) for i in range(system.getNumForces())]
        for f in listforce:
            if type(f) == mm.PeriodicTorsionForce:
                print('PeriodicTorsionForce object is recognized')
                periodictorsionForce = [f for f in listforce if type(f) == mm.PeriodicTorsionForce][0]
            elif type(f) == mm.CMAPTorsionForce:
                print('CMAPTorsionForce object is recognized')
                cmaptorsionForce = [f for f in listforce if type(f) == mm.CMAPTorsionForce][0]
            elif type(f) == mm.CustomTorsionForce:
                print('CustomTorsionForce object is recognized')
                customtorsionForce = [f for f in listforce if type(f) == mm.CustomTorsionForce][0]
            elif type(f) == mm.NonbondedForce:
                print('NonbondedForce object is recognized')
                nonbondedForce = [f for f in listforce if type(f) == mm.NonbondedForce][0]
                self.nonbondedCutoff = nonbondedForce.getCutoffDistance()._value
                self.nonbondedMethod = nonbondedForce.getNonbondedMethod()
                self.dielectric = nonbondedForce.getReactionFieldDielectric()

                if self.nonbondedMethod == 4:
                    alpha_ewald = nonbondedForce.getPMEParameters()[0]._value
                    if alpha_ewald == 0:
                        self.alpha_ewald = (1.0/self.nonbondedCutoff) * sqrt(-log(2.0*nonbondedForce.getEwaldErrorTolerance()))
                    else:
                        self.alpha_ewald == alpha_ewald
                
                self.nbexcpt = [nonbondedForce.getExceptionParameters(i) for i in range(nonbondedForce.getNumExceptions())]
                self.nbparam = [nonbondedForce.getParticleParameters(i)[0]._value for i in range(system.getNumParticles())]
            elif type(f) == mm.CustomNonbondedForce: 
                print('CustomNonbondedForce is recognized')
                logging.getLogger().warning("Warning: LennardJones parameters (sigma/epsilon) will be inherited from customNonbondedForce object, and Electrostatic parameters (charge) will be inherited from NonbondedForce object. Please be aware that such settings are in line with the ForceField definition.")
                self.customnbFlag = True
                customNonbondedForce = [f for f in listforce if type(f) == mm.CustomNonbondedForce][0]
                if customnbforce == 'LennardJonesForce':
                    acoef = customNonbondedForce.getTabulatedFunction(0).getFunctionParameters()
                    bcoef = customNonbondedForce.getTabulatedFunction(1).getFunctionParameters()
                    ljtype = [customNonbondedForce.getParticleParameters(i)[0] for i in range(system.getNumParticles())]
                    self.ljparam = [ljtype, acoef, bcoef]
                    customBondForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == mm.CustomBondForce][0]
                    self.lj14param = []
                    for i in range(customBondForce.getNumBonds()):
                        idx0 = customBondForce.getBondParameters(i)[0]
                        idx1 = customBondForce.getBondParameters(i)[1]
                        sig_14 = customBondForce.getBondParameters(i)[2][0]
                        eps_14 = customBondForce.getBondParameters(i)[2][1]
                        self.lj14param.append((idx0,idx1,0,sig_14,eps_14))
                elif customnbforce == 'SAPTForce':
                    raise NotImplementedError('TODO')
                else:
                    raise ValueError('Option for customNonbondedForce is unknown. Only LennardJonesForce and SAPTForce is currently supported.')
            elif type(f) == mm.DrudeForce:
                print('DrudeForce is recognized')
                DrudeForce = [f for f in listforce if type(f) == mm.DrudeForce][0] 
        
        if not self.customnbFlag:
            logging.getLogger().warning("LennardJones parameters (sigma/epsilon) and Electrostatic parameters (charge) will be both inherited from NonbondedForce object. Please be aware that such settings are in line with the ForceField definition.")
            self.ljparam = [(nonbondedForce.getParticleParameters(i)[1]._value,nonbondedForce.getParticleParameters(i)[2]._value) for i in range(system.getNumParticles())]
    
    def addGroupForce(self, system):
        """Add Group Force in the OpenMM System object"""

        i=0
        for force in self.groupForce:
            forceIndex = system.addForce(force)
            self.groupForceIndex.append(forceIndex)
            print(self.groupForceName[i]+' Force is now added to the system with Force index: '+str(forceIndex))
            i+=1

        return self.groupForceName, self.groupForceIndex
    
    def createPeriodicTorsionForce(self, listSingle, forceName, **args):
        """Create the PeriodicTorsionForce object for the Interaction in listSingle

        Parameters
        ----------
        listSingle : list
            A list that includes the index of atom for computing the torsional energy.
        forceName : str
            A string for the Force object.
        """

        pass
    
    def createCMAPTorsionForce(self, listSingle, forceName, **args):
        """Create the CMAPTorsionForce object for the Interaction in listSingle

        Parameters
        ----------
        listSingle : list
            A list that includes the index of atom for computing the torsional energy.
            This function is only considered in CHARMM type force field.
        forceName : str
            A string for the Force object.
        """

        pass

    def createCustomTorsionForce(self, listSingle, forceName, **args):
        """Create the CustomTorsionForce object for the Interaction in listSingle

        Parameters
        ----------
        listSingle : list
            A list that includes the index of atom for computing the torsional energy.
            This type of torsional energy is usually for modeling improper torsion.
        forceName : str
            A string for the Force object.
        """

        pass

    @staticmethod
    def checkPair(listPair, listDistance):

        if len(listPair) != len(listDistance):
            raise ValueError('The number of distance values and the number of pairs is not equal.')

        for pair in listPair:
            if len(pair) != 2:
                raise ValueError('The number of elements in Pair is not equal to 2.')
    
    def generateExclusion(self, force):
        """Create the exclusions for the Force object. Definition of exclusions will be inherited
        from the nonbondedForce object.

        Parameters
        ----------
        force : Force
            A Force object for exclusion.
        """

        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in self.nbexcpt:
            force.addExclusion(idx1, idx2)

        return force

    def createLennardJonesForce(self, listPair, forceName, scale=1.0, **args):
        """Create the LennardJonesForce object for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """
        
        args['switchDistance'] = None
        
        if self.customnbFlag:
            atomType = self.ljparam[0]
            acoef = self.ljparam[1]
            bcoef = self.ljparam[2]
            numLjTypes = int(acoef[0])
            
            force = mm.CustomNonbondedForce('scale*ljacoef(ljtype1, ljtype2)/r^12 - scale*ljbcoef(ljtype1, ljtype2)/r^6;')
            force.addTabulatedFunction('ljacoef', mm.Discrete2DFunction(numLjTypes, numLjTypes, list(acoef[2])))
            force.addTabulatedFunction('ljbcoef', mm.Discrete2DFunction(numLjTypes, numLjTypes, list(bcoef[2])))
            force.addPerParticleParameter('ljtype')
            force.addGlobalParameter('scale',scale)
            
            # add ghost type (numLjTypes) which does not have any LennardJones interaction
            # this type will be assigned to the particles which are not belonged to the listPair
            tbA = np.zeros((numLjTypes+1)**2,np.double)
            tbB = np.zeros((numLjTypes+1)**2,np.double)
            
            for i in range(numLjTypes):
                for j in range(numLjTypes):
                    tbA[i*(numLjTypes+1)+j] = acoef[2][i*numLjTypes+j]
                    tbB[i*(numLjTypes+1)+j] = bcoef[2][i*numLjTypes+j]
            
            # exclude self-interaction for listPair[0] and listPair[1]
            # This part should be debugged if listPair[0] and listPair[1] both contain the common ljtype
            if listPair[0] != listPair[1]:
                listPairType = [[atomType[i] for i in listPair[0]],[atomType[i] for i in listPair[1]]]
                selfPair = set(product(listPairType[0],repeat=2)).union(set(product(listPairType[1],repeat=2)))

                for (i,j) in selfPair:
                    tbA[int(i)*(numLjTypes+1)+int(j)] = 0
                    tbB[int(i)*(numLjTypes+1)+int(j)] = 0
                
                #if not (tbA==tbA.T).all() or (tbB==tbB.T).all():
                    #raise ValueError('Coefficient matrix for LennardJones interaction is not symmetric')
            
            force.getTabulatedFunction(0).setFunctionParameters(numLjTypes+1,numLjTypes+1,tbA)
            force.getTabulatedFunction(1).setFunctionParameters(numLjTypes+1,numLjTypes+1,tbB)
             
            i=0
            for atom in atomType:
                if (i in listPair[0]) or (i in listPair[1]):
                    force.addParticle((atom,))
                else:
                    force.addParticle((numLjTypes,))
                i+=1
        else:
            expr = "4*scale*epsilon*((sigma/r)^12 - (sigma/r)^6);"
            expr += "epsilon = sqrt(epsilon1*epsilon2);"
            expr += "sigma = 0.5*(sigma1+sigma2);" 
            expr += "scale = {:f};".format(scale)
            
            force = mm.CustomNonbondedForce(expr)
            force.addPerParticleParameter('sigma')
            force.addPerParticleParameter('epsilon')
            
            for atom in self.ljparam:
                force.addParticle([atom[0],atom[1]])
            
            force.addInteractionGroup(listPair[0], listPair[1])

        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force.setUseLongRangeCorrection(self.DispersionCorrection)
        force.setCutoffDistance(self.nonbondedCutoff)
        
        force = self.generateExclusion(force)
        
        self.groupForce.append(force)
        self.groupForceName.append(forceName)
         
        if self.customnbFlag:
            Exclparam = self.lj14param
        else:
            Exclparam = self.nbexcpt

        listExcl = []
        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in Exclparam:
            if ((idx1 in listPair[0]) and (idx2 in listPair[1])) or ((idx1 in listPair[1]) and (idx2 in listPair[0])):
                listExcl.append([idx1,idx2,q_excpt,sig_excpt,eps_excpt])

        if not listExcl:
            return force
        else:
            print('Exclusions for '+forceName+' are now considered.')
            
            exclexpr = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"

            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('sigma')
            exclforce.addPerBondParameter('epsilon')

            for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in Exclparam:
                exclforce.addBond(idx1, idx2, [sig_excpt, eps_excpt])

            self.groupForce.append(exclforce)
            self.groupForceName.append(forceName+'_excl')

            return force, exclforce

    def plotLennardJonesForce(self, listPair, **args):
        pass
    
    def computeLennardJonesPairForce(self, listPair, listDistance, scale=1.0, verbose=False):
        
        self.checkPair(listPair, listDistance)

        if self.customnbFlag:
            listA     = []
            listB     = []
            atomType  = self.ljparam[0]
            acoef     = self.ljparam[1]
            bcoef     = self.ljparam[2]
            numLjTypes = int(acoef[0])
            for i in range(len(listDistance)):
                ljtypeA = int(atomType[listPair[i][0]])
                ljtypeB = int(atomType[listPair[i][1]])
                listA.append(acoef[2][ljtypeA*numLjTypes+ljtypeB])
                listB.append(bcoef[2][ljtypeA*numLjTypes+ljtypeB])
            energy = scale*sum(map(lambda A,B,r: A/r**12 - B/r**6,listA,listB,listDistance))
        else:
            listsigpair = []
            listepspair = []
            for i in range(len(listDistance)):
                sigpair = 0.5*(self.ljparam[listPair[i][0]][0]+self.ljparam[listPair[i][1]][0])
                epspair = sqrt(self.ljparam[listPair[i][0]][1]*self.ljparam[listPair[i][1]][1])
                listsigpair.append(sigpair)
                listepspair.append(epspair)
            energy = scale*sum(map(lambda sig,eps,r: 4*eps*((sig/r)**12 - (sig/r)**6),listsigpair,listepspair,listDistance))
        
        if verbose:
            print('LennardJones Energy is evaluated as: '+str(energy)+' (kJ/mol)')
 
        return energy
    
    def computeCoulombNoCutoffPairForce(self, listPair, listDistance, scale=1.0, verbose=False):
        
        self.checkPair(listPair, listDistance)

        listchargeprod  = []
        for i in range(len(listDistance)):
            chargeprod = self.nbparam[listPair[i][0]]*self.nbparam[listPair[i][1]]
            listchargeprod.append(chargeprod)
        energy = scale*sum(map(lambda charge,r: InteractionGroup.ONE_4PI_EPS0*charge/r,listchargeprod,listDistance))

        if verbose:
            print('CoulombNoCutOff Energy is evaluated as: '+str(energy)+' (kJ/mol)')

        return energy
    
    def computeCoulombCutoffPairForce(self, listPair, listDistance, scale=1.0, verbose=False):

        self.checkPair(listPair, listDistance)
        
        k = (1/self.nonbondedCutoff)**3 * ((self.dielectric-1)/(2*self.dielectric+1))
        c = (1/self.nonbondedCutoff) * ((3*self.dielectric)/(2*self.dielectric+1))
        
        listchargeprod  = []
        for i in range(len(listDistance)):
            chargeprod = self.nbparam[listPair[i][0]]*self.nbparam[listPair[i][1]]
            listchargeprod.append(chargeprod)
        energy = scale*sum(map(lambda charge,r: InteractionGroup.ONE_4PI_EPS0*charge*(1/r+k*r**2-c),listchargeprod,listDistance))
        
        if verbose:
            print('CoulombCutOff Energy is evaluated as: '+str(energy)+' (kJ/mol)')

        return energy

    def computePMEPairForce(self, listPair, listDistance, scale=1.0, verbose=False):

        self.checkPair(listPair, listDistance)

        listchargeprod  = []
        for i in range(len(listDistance)):
            chargeprod = self.nbparam[listPair[i][0]]*self.nbparam[listPair[i][1]]
            listchargeprod.append(chargeprod)
        energy = scale*sum(map(lambda charge,r: InteractionGroup.ONE_4PI_EPS0*charge*erfc(self.alpha_ewald*r)/r,listchargeprod,listDistance))
        
        if verbose:
            print('PME (Direct Space) Energy is evaluated as: '+str(energy)+' (kJ/mol)')

        return energy

    def createPMEForce(self, listPair, forceName, scale=1.0, **args):
        """Create the PME-Direct Space Force object for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """
 
        args['switchDistance'] = None

        expr = "scale*ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*r)/r;"
        expr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
        expr += "chargeprod = charge1*charge2;"
        expr += "alpha_ewald = {:f};".format(self.alpha_ewald)
        expr += "scale = {:f};".format(scale)

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('charge')
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force.setUseSwitchingFunction(False)
        force.setUseLongRangeCorrection(False)
        force.setCutoffDistance(self.nonbondedCutoff)

        for atom in self.nbparam:
            force.addParticle([atom])
        
        force = self.generateExclusion(force)

        force.addInteractionGroup(listPair[0], listPair[1])
        self.groupForce.append(force)
        self.groupForceName.append(forceName)
        
        listExcl = []
        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in self.nbexcpt:
            if ((idx1 in listPair[0]) and (idx2 in listPair[1])) or ((idx1 in listPair[1]) and (idx2 in listPair[0])):
                listExcl.append([idx1,idx2,q_excpt,sig_excpt,eps_excpt])

        if not listExcl:
            return force
        else:
            print('Exclusions for '+forceName+' are now considered.')

            exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r - ONE_4PI_EPS0*chargeprod_nat*erf(alpha_ewald*r)/r;"
            exclexpr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
            exclexpr += "alpha_ewald = {:f};".format(self.alpha_ewald)
            
            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('chargeprod_excl')
            exclforce.addPerBondParameter('chargeprod_nat')
            
            for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in listExcl:
                q1 = self.nbparam[idx1]
                q2 = self.nbparam[idx2]
                exclforce.addBond(idx1, idx2, [q_excpt, q1*q2])
            
            self.groupForce.append(exclforce)
            self.groupForceName.append(forceName+'_excl')
                    
            return force, exclforce

    def createCoulombNoCutoffForce(self, listPair, forceName, scale=1.0, **args):
        """Create the CoulombForce object without CutOff for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """

        args['switchDistance'] = None

        expr = "scale*ONE_4PI_EPS0*chargeprod/r;"
        expr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
        expr += "chargeprod = charge1*charge2;"
        expr += "scale = {:f};".format(scale)

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('charge')
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force.setUseSwitchingFunction(False)
        force.setUseLongRangeCorrection(False)
        force.setCutoffDistance(self.nonbondedCutoff)

        for atom in self.nbparam:
            force.addParticle([atom])
        
        force = self.generateExclusion(force)

        force.addInteractionGroup(listPair[0], listPair[1])
        self.groupForce.append(force)
        self.groupForceName.append(forceName)

        listExcl = []
        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in self.nbexcpt:
            if ((idx1 in listPair[0]) and (idx2 in listPair[1])) or ((idx1 in listPair[1]) and (idx2 in listPair[0])):
                listExcl.append([idx1,idx2,q_excpt,sig_excpt,eps_excpt])

        if not listExcl:
            return force
        else:
            print('Exclusions for '+forceName+' are now considered.')

            exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r;"
            exclexpr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
            exclexpr += "alpha_ewald = {:f};".format(self.alpha_ewald)

            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('chargeprod_excl')

            for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in listExcl:
                q1 = self.nbparam[idx1]
                q2 = self.nbparam[idx2]
                exclforce.addBond(idx1, idx2, [q_excpt])

            self.groupForce.append(exclforce)
            self.groupForceName.append(forceName+'_excl')

            return force, exclforce

    def createCoulombCutoffForce(self, listPair, forceName, scale=1.0, **args):
        """Create the CoulombForce object with CutOff for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """

        args['switchDistance'] = None

        expr = "scale*ONE_4PI_EPS0*chargeprod*(1/r+k*r^2-c);"
        expr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
        expr += "k = {:f};".format((1/self.nonbondedCutoff)**3 * ((self.dielectric-1)/(2*self.dielectric+1)))
        expr += "c = {:f};".format((1/self.nonbondedCutoff) * ((3*self.dielectric)/(2*self.dielectric+1)))
        expr += "chargeprod = charge1*charge2;"
        expr += "scale = {:f};".format(scale)

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('charge')
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force.setUseSwitchingFunction(False)
        force.setUseLongRangeCorrection(False)
        force.setCutoffDistance(self.nonbondedCutoff)

        for atom in self.nbparam:
            force.addParticle([atom])
        
        force = self.generateExclusion(force)
        
        force.addInteractionGroup(listPair[0], listPair[1])
        self.groupForce.append(force)
        self.groupForceName.append(forceName)
        
        listExcl = []
        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in self.nbexcpt:
            if ((idx1 in listPair[0]) and (idx2 in listPair[1])) or ((idx1 in listPair[1]) and (idx2 in listPair[0])):
                listExcl.append([idx1,idx2,q_excpt,sig_excpt,eps_excpt])

        if not listExcl:
            return force
        else:
            print('Exclusions for '+forceName+' are now considered.')

            exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r;"
            exclexpr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
            exclexpr += "alpha_ewald = {:f};".format(self.alpha_ewald)

            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('chargeprod_excl')

            for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in listExcl:
                q1 = self.nbparam[idx1]
                q2 = self.nbparam[idx2]
                exclforce.addBond(idx1, idx2, [q_excpt])

            self.groupForce.append(exclforce)
            self.groupForceName.append(forceName+'_excl')

            return force, exclforce

    def createSAPTForce(self, listPair, forceName, **args):
        pass
