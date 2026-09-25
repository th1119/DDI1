import numpy as np
from rdkit import Chem


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, use_chirality=True, explicit_H=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
         'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',  # H?
         'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Mn', 'Other']) + one_of_k_encoding(atom.GetDegree(),
                                                                           [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2, 'Other'
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_length_approximation(bond_type):
    bond_length_dict = {"SINGLE": 1.0, "DOUBLE": 1.4, "TRIPLE": 1.8, "AROMATIC": 1.5}
    return bond_length_dict.get(bond_type, 1.0)


def encode_bond_15(bond):
    # 7+4+2+2 = 15
    bond_dir = [0] * 7
    bond_dir[bond.GetBondDir()] = 1

    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_type = [1, 0, 0, 0]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_type = [0, 1, 0, 0]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_type = [0, 0, 1, 0]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_type = [0, 0, 0, 1]
    else:
        bond_type = [0, 0, 0, 0]

    bond_length = bond_length_approximation(bond.GetBondType())

    in_ring = [0, 0]
    in_ring[int(bond.IsInRing())] = 1

    edge_encode = bond_dir + bond_type + [bond_length, bond_length ** 2] + in_ring

    return np.array(edge_encode)
