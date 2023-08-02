import simtk.openmm as mm
import simtk.openmm.app as app
import numpy as np
from .constants import ONE_4PI_EPS0

def generateExclusion(self, force, forceName, param):
    """Create the exclusions for the Force object. Definition of exclusions will be inherited
    from the nonbondedForce object."""

    for (idx1, idx2, q_excl, sig_excl, eps_excl) in param['nbexclparam']:
        force.addExclusion(idx1, idx2)

    return force

def checkExclusion(self, listPair, param):
    """Check the exclusions in nonbondedForce object are overlapped with the atom in listPair"""

    listexcl = []
    for (idx1, idx2, q_excl, sig_excl, eps_excl) in param['nbexclparam']:
        if ((idx1 in listPair[0]) and (idx2 in listPair[1])) or ((idx1 in listPair[1]) and (idx2 in listPair[0])):
            listexcl.append([idx1,idx2,q_excl,sig_excl,eps_excl])

    return listexcl

def createExclusionBondForce(self, listPair, forceName, method, param, tag=''):
    """Create the exclusion bonds for the CustomBondForce object. Usually 1-4 bond is selected."""

    listexcl = self.checkExclusion(listPair, param)

    if not listexcl:
        return None
    else:
        print('Exclusions for '+forceName+' are now considered.')

        if tag == 'PME':
            exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r - ONE_4PI_EPS0*chargeprod_nat*erf(alpha_ewald*r)/r;"
            exclexpr += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
            exclexpr += "alpha_ewald = {:f};".format(method['alpha_ewald'])

            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('chargeprod_excl')
            exclforce.addPerBondParameter('chargeprod_nat')

            for (idx1, idx2, q_excl, sig_excl, eps_excl) in param['nbexclparam']:
                q1 = param['elecparam'][idx1]
                q2 = param['elecparam'][idx2]
                exclforce.addBond(idx1, idx2, [q_excl, q1*q2])

        elif tag in ['NoCutoff','Cutoff']:
            exclexpr = "ONE_4PI_EPS0*chargeprod_excl/r;"
            exclexpr += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)

            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('chargeprod_excl')

            for (idx1, idx2, q_excl, sig_excl, eps_excl) in param['nbexclparam']:
                exclforce.addBond(idx1, idx2, [q_excl])

        elif tag == 'LennardJones':
            exclexpr = "4*epsilon*((sigma/r)^12 - (sigma/r)^6);"

            exclforce = mm.CustomBondForce(exclexpr)
            exclforce.addPerBondParameter('sigma')
            exclforce.addPerBondParameter('epsilon')

            for (idx1, idx2, q_excl, sig_excl, eps_excl) in param['ljexclparam']:
                exclforce.addBond(idx1, idx2, [sig_excl, eps_excl])

        self.groupForce.append(exclforce)
        self.groupForceName.append(forceName+'_excl')

        return exclforce
