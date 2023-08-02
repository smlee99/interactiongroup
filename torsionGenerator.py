import simtk.openmm as mm
import simtk.openmm.app as app
import numpy as np
from .utils import *

class addPeriodicTorsionForce(object):

    def __init__(self, group, listTorsion, forceName):
        self.tag = 'PeriodicTorsion'
        self.method = group.method
        self.param = group.param

        sanitycheckList(listTorsion, self.tag)
        self.createForce(listTorsion, forceName)

    def createForce(self, listTorsion, forceName):
        """Create the PeriodicTorsionForce object for the Interaction in listSingle

        Parameters
        ----------
        listTorsion : list
            A list that includes the index of atom for computing the torsional energy.
        forceName : str
            A string for the Force object.
        """


        return force


