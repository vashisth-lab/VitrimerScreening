from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
import torch
from torch_geometric.data import Data


possible_atom_list = ['C', 'N', 'O', '*']
# possible_atom_list = ['Br', 'C', 'Cl', 'F', 'Li', 'N', 'O', 'P', 'S', 'Si', '*']
possible_numH_list= [0, 1, 2, 3, 4, '>4']
possible_valence_list = [0, 1, 2, 3, 4, '>4']
possible_degree_list = [0, 1, 2, 3, 4, '>4']
possible_bond_list = [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, '*']


def onehot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_featurization(atom):
    atom_feats = np.array(onehot_encoding_unk(atom.GetSymbol(), possible_atom_list)
    + onehot_encoding_unk(atom.GetTotalNumHs(), possible_numH_list)
    + onehot_encoding_unk(atom.GetImplicitValence(), possible_valence_list) 
    + onehot_encoding_unk(atom.GetDegree(), possible_degree_list) 
    + [atom.GetIsAromatic()])
    return atom_feats


def bond_pairs_between_atoms(mol):                                   
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def bond_featurization(bond):
    bond_feats = np.array(onehot_encoding_unk(bond.GetBondType(), possible_bond_list)
    + [bond.GetIsConjugated()]
    + [bond.IsInRing()])
    return bond_feats


def mol_to_graph(mol):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    edge_attr = []
    node_features= [atom_featurization(atom) for atom in atoms]
    edge_index = bond_pairs_between_atoms(mol)
    for bond in bonds:
        edge_attr.append(bond_featurization(bond))
        edge_attr.append(bond_featurization(bond))
    data = Data(x=torch.tensor(np.array(node_features), dtype=torch.float),
                edge_index=torch.tensor(np.array(edge_index), dtype=torch.long), edge_attr=torch.tensor(np.array(edge_attr), dtype=torch.float))
    return data

