from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from torch_geometric.data.collate import collate
from gnn_utils import mol_to_graph
import torch, sys, os


def vitrimerize(acid, epoxide):
    acid_mol = Chem.MolFromSmiles(acid)
    epoxide_mol = Chem.MolFromSmiles(epoxide)
    rxn1 = AllChem.ReactionFromSmarts('[CX3:1](=O)[OX2H1:2]>>[CX3:1](=O)[OX2:2][*]')
    acid_mol = rxn1.RunReactants((acid_mol, ))[0][0]
    Chem.SanitizeMol(acid_mol)
    rxn2 = AllChem.ReactionFromSmarts('[OD2r3:1]1[#6D2r3:2][#6r3:3]1>>[#6:3]([OD2:1])[#6D2:2][*]')
    epoxide_mol = rxn2.RunReactants((epoxide_mol, ))[0][0]
    Chem.SanitizeMol(epoxide_mol)
    rxn = AllChem.ReactionFromSmarts('[CX3:1](=O)[OX2H1:2].[OD2r3:3]1[#6D2r3:4][#6r3:5]1>>[CX3:1](=O)[OX2:2][#6D2:4][#6:5]([OD2:3])')
    vitrimer_mol = rxn.RunReactants((acid_mol, epoxide_mol))[0][0]
    Chem.SanitizeMol(vitrimer_mol)
    vitrimer = Chem.MolToSmiles(vitrimer_mol)
    return Chem.CanonSmiles(vitrimer)


def collate_data(data_list):
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    return data, slices


def get_labeled_graph(file):
    acid = pd.read_csv('data/' + file + '.csv')['acid'].to_list()
    epoxide = pd.read_csv('data/' + file + '.csv')['epoxide'].to_list()
    tg = pd.read_csv('data/' + file + '.csv')['tg'].to_list()
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    data_list = [mol_to_graph(mol) for mol in mols]
    
    for i, object in enumerate(data_list):
        object.y = torch.tensor([tg[i]], dtype=torch.float)

    data, slices = collate_data(data_list)
    torch.save((data, slices), 'data/%s_graph.pt' % (file))


def get_unlabeled_graph(file, i_batch):
    acid = pd.read_csv('data/' + file + '.csv')['acid'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    epoxide = pd.read_csv('data/' + file + '.csv')['epoxide'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    data_list = [mol_to_graph(mol) for mol in mols]

    data, slices = collate_data(data_list)
    torch.save((data, slices), f'data/{file}_graph/{file}_graph_{i_batch:d}.pt')


get_labeled_graph('labeled_train')
get_labeled_graph('labeled_test')

file = 'unlabeled'
os.mkdir(f'data/{file}_graph')
get_unlabeled_graph(file, int(sys.argv[1]))

file = 'unlabeled_synthesis'
os.mkdir(f'data/{file}_graph')
get_unlabeled_graph(file, int(sys.argv[1]))
