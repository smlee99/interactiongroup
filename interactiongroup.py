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

    def __init__(self, system, customnbforce=''):
        """Load the system object and create general nonbonded force settings.

        Parameters
        ----------
        system : OpenMM system
            A system object created by the forcefield. Note that defining the InteractionGroup
            object should be done after all forces and exclusions/exceptions are made. General
            nonbonded force settings (i.e., cutoff distance) are then be extracted from the
            nonbondedForce and (if exist) customNonbondedForce in system object.
        customnbforce : customNonbondedForce type
            Option for the customNonbondedForce. This is generally regarded as the LennardJones
            force, including the NBFIX one. Note that there are various types of forcefield which
            also define the customNonbondedForce but with different analytic expression. This
            module currently supports only the LennardJonesForce.
        """
        
        self.customnbFlag = False
        self.groupForce = []
        self.groupForceName = []
        self.groupForceIndex = []
        
        self.registerForce(system, customnbforce)

    def registerForce(self, system, customnbforce):
        """Register Force option in the OpenMM System object"""
        
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
                
                self.nbDispersionCorrection = nonbondedForce.getUseDispersionCorrection()
                self.nbSwitchingFunction = nonbondedForce.getUseSwitchingFunction()
                
                if self.nbSwitchingFunction:
                    self.nbSwitchingDistance = nonbondedForce.getSwitchingDistance()

                if self.nonbondedMethod == 4:
                    alpha_ewald = nonbondedForce.getPMEParameters()[0]._value

                    if alpha_ewald == 0:
                        self.alpha_ewald = (1.0/self.nonbondedCutoff) * sqrt(-log(2.0*nonbondedForce.getEwaldErrorTolerance()))
                    else:
                        self.alpha_ewald == alpha_ewald

                self.nbexcpt = [nonbondedForce.getExceptionParameters(i) for i in range(nonbondedForce.getNumExceptions())]
                self.nbparam = [nonbondedForce.getParticleParameters(i)[0]._value for i in range(system.getNumParticles())]
                
            elif type(f) == mm.CustomNonbondedForce:
                print('CustomNonbondedForce is recognized as a '+customnbforce)
                logging.getLogger().warning("Warning: LennardJones parameters (sigma/epsilon) will be inherited from customNonbondedForce object, and Electrostatic parameters (charge) will be inherited from NonbondedForce object. Please be aware that such settings are in line with the ForceField definition.")
                self.customnbFlag = True
                customNonbondedForce = [f for f in listforce if type(f) == mm.CustomNonbondedForce][0]
                
                self.customnbDispersionCorrection = customNonbondedForce.getUseDispersionCorrection()
                self.customnbSwitchingFunction = customNonbondedForce.getUseSwitchingFunction()

                if self.customnbSwitchingFunction:
                    self.customnbSwitchingDistance = customNonbondedForce.getSwitchingDistance()

                if customnbforce == 'LennardJonesForce': 
                    acoef = customNonbondedForce.getTabulatedFunction(0).getFunctionParameters()
                    bcoef = customNonbondedForce.getTabulatedFunction(1).getFunctionParameters()
                    ljtype = [customNonbondedForce.getParticleParameters(i)[0] for i in range(system.getNumParticles())]
                    self.ljparam = [ljtype, acoef, bcoef]

                elif customnbforce == 'SAPTForce':
                    self.ljparam = []
                    for i in range(system.getNumParticles()):
                        self.ljparam.append(customNonbondedForce.getParticleParameters(i))
                
                else:
                    raise AssertionError('Unrecognized customNonbondedForce object [%s]' % customnbforce)

                customBondForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == mm.CustomBondForce]

                if customBondForce:
                    self.lj14param = []
                    for i in range(customBondForce[0].getNumBonds()):
                        idx0 = customBondForce[0].getBondParameters(i)[0]
                        idx1 = customBondForce[0].getBondParameters(i)[1]
                        sig_14 = customBondForce[0].getBondParameters(i)[2][0]
                        eps_14 = customBondForce[0].getBondParameters(i)[2][1]
                        self.lj14param.append((idx0,idx1,0,sig_14,eps_14))

            elif type(f) == mm.DrudeForce:
                print('DrudeForce is recognized')
                DrudeForce = [f for f in listforce if type(f) == mm.DrudeForce][0]

        if not self.customnbFlag:
            logging.getLogger().warning("LennardJones parameters (sigma/epsilon) and Electrostatic parameters (charge) will be both inherited from NonbondedForce object. Please be aware that such settings are in line with the ForceField definition.")
            self.ljparam = [(nonbondedForce.getParticleParameters(i)[1]._value,nonbondedForce.getParticleParameters(i)[2]._value) for i in range(system.getNumParticles())]

        return
    
    @staticmethod
    def checkList(listAtom, tag=''):
        """Validate the listPair."""
        
        if tag == 'bond':
            for doublet in listAtom:
                if len(doublet) != 2:
                    raise ValueError('The number of atom groups in listBond is not equal to 2.') 
        elif tag == 'angle':
            for triplet in listAtom:
                if len(triplet) != 3:
                    raise ValueError('The number of atom groups in listAngle is not equal to 2.') 
        elif tag == 'torsion':
            for quartet in listAtom:
                if len(quartet) != 4:
                    raise ValueError('The number of atom groups in listTorsion is not equal to 4.')
        elif tag in ['LennardJones', 'PME', 'Cutoff', 'NoCutoff', 'SAPT']:
            if len(pair) != 2:
                raise ValueError('The number of atom groups in listPair is not equal to 2.')

    def addGroupForce(self, system):
        """Add Group Force in the OpenMM System object"""

        i=0
        for force in self.groupForce:
            forceIndex = system.addForce(force)
            self.groupForceIndex.append(forceIndex)
            print(self.groupForceName[i]+' Force is now added to the system with Force index: '+str(forceIndex))
            i+=1

        return self.groupForceName, self.groupForceIndex
    
    def generateExclusion(self, force, forceName):
        """Create the exclusions for the Force object. Definition of exclusions will be inherited
        from the nonbondedForce object."""

        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in self.nbexcpt:
            force.addExclusion(idx1, idx2)

        return force
    
    def createExclusionBondForce(self, listPair, forceName, exclparam, tag=''):
        """Create the exclusion bonds for the CustomBondForce object. Usually 1-4 bond is selected."""
        
        listexcl = []
        for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in self.nbexcpt:
            if ((idx1 in listPair[0]) and (idx2 in listPair[1])) or ((idx1 in listPair[1]) and (idx2 in listPair[0])):
                listexcl.append([idx1,idx2,q_excpt,sig_excpt,eps_excpt])

        if not listexcl:
            return None
        else:
            print('Exclusions for '+forceName+' are now considered.')
            
            if tag == 'PME':
                exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r - ONE_4PI_EPS0*chargeprod_nat*erf(alpha_ewald*r)/r;"
                exclexpr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
                exclexpr += "alpha_ewald = {:f};".format(self.alpha_ewald)

                exclforce = mm.CustomBondForce(exclexpr)
                exclforce.addPerBondParameter('chargeprod_excl')
                exclforce.addPerBondParameter('chargeprod_nat')

                for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in exclparam:
                    q1 = self.nbparam[idx1]
                    q2 = self.nbparam[idx2]
                    exclforce.addBond(idx1, idx2, [q_excpt, q1*q2])

            elif tag in ['NoCutoff','Cutoff']:
                exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r;"
                exclexpr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)

                exclforce = mm.CustomBondForce(exclexpr)
                exclforce.addPerBondParameter('chargeprod_excl')

                for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in exclparam:
                    q1 = self.nbparam[idx1]
                    q2 = self.nbparam[idx2]
                    exclforce.addBond(idx1, idx2, [q_excpt])

            elif tag == 'LennardJones':
                exclexpr = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"

                exclforce = mm.CustomBondForce(exclexpr)
                exclforce.addPerBondParameter('sigma')
                exclforce.addPerBondParameter('epsilon')

                for (idx1, idx2, q_excpt, sig_excpt, eps_excpt) in exclparam: 
                    exclforce.addBond(idx1, idx2, [sig_excpt, eps_excpt])

            self.groupForce.append(exclforce)
            self.groupForceName.append(forceName+'_excl')
            
            else:
                return None

            return exclforce

    def postprocessForce(self, force, forceName, tag=''):
        """Set the nonbonded options for the Force object and create the exclusions."""

        if tag in ['PME','Cutoff','NoCutoff']:
            force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(self.nonbondedCutoff)

            force.setUseLongRangeCorrection(False)

            if self.nbSwitchingFunction:
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(self.nbSwitchingDistance)

        elif tag == 'LennardJones':
            force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
            force.setCutoffDistance(self.nonbondedCutoff)

            force.setUseLongRangeCorrection(self.customnbDispersionCorrection)

            if self.customnbSwitchingFunction:
                force.setUseSwitchingFunction(True)
                force.setSwitchingDistance(self.customnbSwitchingDistance)
        
        force = self.generateExclusion(force)
        
        self.groupForce.append(force)
        self.groupForceName.append(forceName)

        return force, forceName

    def createPeriodicTorsionForce(self, listSingle, forceName, tag='PeriodicTorsion'):
        """Create the PeriodicTorsionForce object for the Interaction in listSingle

        Parameters
        ----------
        listSingle : list
            A list that includes the index of atom for computing the torsional energy.
        forceName : str
            A string for the Force object.
        """

        pass

    def createLennardJonesForce(self, listPair, forceName, scale=1.0, tag='LennardJones'):
        """Create the LennardJonesForce object for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """
        
        self.checkList(listPair, tag)

        if self.customnbFlag:
            exclparam = self.lj14param

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
            exclparam = self.nbexcpt

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
        
        force = postprocessForce(force, forceName, tag)
        
        exclforce = self.createExclusionBondForce(listPair, forceName, exclparam, tag)
        
        if not exclforce:
            return force
        else:
            return force, exclforce

    def createPMERealSpaceForce(self, listPair, forceName, scale=1.0, tag='PME'):
        """Create the PME-Direct Space Force object for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """
        
        self.checkList(listPair, tag)

        expr = "scale*ONE_4PI_EPS0*chargeprod*erfc(alpha_ewald*r)/r;"
        expr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
        expr += "chargeprod = charge1*charge2;"
        expr += "alpha_ewald = {:f};".format(self.alpha_ewald)
        expr += "scale = {:f};".format(scale)

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('charge')
            force.addParticle([atom])
       
        force.addInteractionGroup(listPair[0], listPair[1])

        force = postprocessForce(force, forceName, tag)
        
        exclparam = self.nbexcpt
        exclforce = self.createExclusionBondForce(listPair, forceName, exclparam, tag)

        if not exclforce:
            return force
        else:
            return force, exclforce

    def createCoulombNoCutoffForce(self, listPair, forceName, scale=1.0, tag='NoCutoff'):
        """Create the CoulombForce object without CutOff for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """
        
        self.checkList(listPair, tag)

        expr = "scale*ONE_4PI_EPS0*chargeprod/r;"
        expr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
        expr += "chargeprod = charge1*charge2;"
        expr += "scale = {:f};".format(scale)

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('charge')

        for atom in self.nbparam:
            force.addParticle([atom])
        
        force.addInteractionGroup(listPair[0], listPair[1])
        
        force = postprocessForce(force, forceName, tag)
        
        exclparam = self.nbexcpt
        exclforce = self.createExclusionBondForce(listPair, forceName, exclparam, tag)

        if not exclforce:
            return force
        else:
            return force, exclforce

    def createCoulombCutoffForce(self, listPair, forceName, scale=1.0, tag='Cutoff'):
        """Create the CoulombForce object with CutOff for the Interaction between listPair

        Parameters
        ----------
        listPair : list
            A list that includes two group list where each includes the atom index of each group
            atom. Note that same group can be assigned for the force computation.
        forceName : str
            A string for the Force object.
        """
        
        self.checkList(listPair, tag)

        expr = "scale*ONE_4PI_EPS0*chargeprod*(1/r+k*r^2-c);"
        expr += "ONE_4PI_EPS0 = {:f};".format(InteractionGroup.ONE_4PI_EPS0)
        expr += "k = {:f};".format((1/self.nonbondedCutoff)**3 * ((self.dielectric-1)/(2*self.dielectric+1)))
        expr += "c = {:f};".format((1/self.nonbondedCutoff) * ((3*self.dielectric)/(2*self.dielectric+1)))
        expr += "chargeprod = charge1*charge2;"
        expr += "scale = {:f};".format(scale)

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('charge')

        for atom in self.nbparam:
            force.addParticle([atom])
        
        force.addInteractionGroup(listPair[0], listPair[1])

        force = postprocessForce(force, forceName, tag)
        
        exclparam = self.nbexcpt
        exclforce = self.createExclusionBondForce(listPair, forceName, exclparam, tag)
        
        if not exclforce:
            return force
        else:
            return force, exclforce

    def createSAPTForce(self, listPair, forceName, scale=1.0, tag='SAPT'):

        self.checkList(listPair, tag)
        
        expr = "(A*exBr - f6*C6/(r^6) - f8*C8/(r^8) - f10*C10/(r^10) - f12*C12/(r^12);"
        expr += "A=Aex-Ael-Ain-Adh;"
        expr += "Aex=sqrt(Aexch1*Aexch2); Ael=sqrt(Aelec1*Aelec2); Ain=sqrt(Aind1*Aind2); Adh=sqrt(Adhf1*Adhf2);"
        expr += "f12 = f10 - exBr*( (1/39916800)*(Br^11)*(1 + Br/12) );"
        expr += "f10 = f8 - exBr*( (1/362880)*(Br^9)*(1 + Br/10 ) );"
        expr += "f8 = f6 - exBr*( (1/5040)*(Br^7)*(1 + Br/8 ) );"
        expr += "f6 = 1 - exBr*(1 + Br * (1 + (1/2)*Br*(1 + (1/3)*Br*(1 + (1/4)*Br*(1 + (1/5)*Br*(1 + (1/6)*Br ) ) )  ) ) );"
        expr += "exBr = exp(-Br);"
        expr += "Br = B*r;"
        expr += "B=(Bexp1+Bexp2)*Bexp1*Bexp2/(Bexp1^2 + Bexp2^2);"
        expr += "C6=sqrt(C61*C62); C8=sqrt(C81*C82); C10=sqrt(C101*C102); C12=sqrt(C121*C122)"

        force = mm.CustomNonbondedForce(expr)
        force.addPerParticleParameter('Aexch')
        force.addPerParticleParameter('Aelec')
        force.addPerParticleParameter('Aind')
        force.addPerParticleParameter('Adhf')
        force.addPerParticleParameter('Bexp')
        force.addPerParticleParameter('C6')
        force.addPerParticleParameter('C8')
        force.addPerParticleParameter('C10')
        force.addPerParticleParameter('C12')

        for atom in self.ljparam:
            force.addParticle([atom])

        force.addInteractionGroup(listPair[0], listPair[1])

        force = postprocessForce(force, forceName, tag)

        exclparam = self.nbexcpt
        exclforce = self.createExclusionBondForce(listPair, forceName, exclparam, tag)

        if not exclforce:
            return force
        else:
            return force, exclforce
