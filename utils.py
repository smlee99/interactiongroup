import numpy as np

def sanitycheckList(listAtom, tag=''):
    """Validate the listPair."""

    if tag == 'bond':
        for doublet in listAtom:
            if len(doublet) != 2:
                raise ValueError('The number of atom groups in listBond is not equal to 2.')
    elif tag == 'angle':
        for triplet in listAtom:
            if len(triplet) != 3:
                raise ValueError('The number of atom groups in listAngle is not equal to 2.')
    elif tag in ['PeriodicTorsion', 'RBTorsion', 'CMAPTorsion']:
        for quartet in listAtom:
            if len(quartet) != 4:
                raise ValueError('The number of atom groups in listTorsion is not equal to 4.')
    elif tag in ['LennardJones', 'PME', 'Cutoff', 'NoCutoff', 'SAPT']:
        if len(pair) != 2:
            raise ValueError('The number of atom groups in listPair is not equal to 2.')

    return
