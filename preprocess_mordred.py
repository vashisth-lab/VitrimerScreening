from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
import sys, os, shutil
from tqdm import tqdm


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
    

def get_labeled_mordred(file):
    calc = Calculator(descriptors, ignore_3D=True)

    acid = pd.read_csv(f'data/{file}.csv')['acid'].to_list()
    epoxide = pd.read_csv(f'data/{file}.csv')['epoxide'].to_list()
    tg = pd.read_csv(f'data/{file}.csv')['tg'].to_list()
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    desc = [calc(m) for m in mols]
    df_desc = pd.DataFrame([d.asdict() for d in desc])

    d = {'acid': acid, 'epoxide': epoxide, 'tg': tg}
    for col in df_desc:
        d[col] = df_desc[col].to_list()
    pd.DataFrame(d).to_csv(f'data/{file}_mordred.csv', index=False, float_format='%.4f')


def get_unlabeled_mordred(file, i_batch):
    calc = Calculator(descriptors, ignore_3D=True)

    acid = pd.read_csv(f'data/{file}.csv')['acid'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    epoxide = pd.read_csv(f'data/{file}.csv')['epoxide'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    desc = [calc(m) for m in mols]
    df_desc = pd.DataFrame([d.asdict() for d in desc])

    d = {'acid': acid, 'epoxide': epoxide}
    for col in df_desc:
        d[col] = df_desc[col].to_list()
    pd.DataFrame(d).to_csv(f'data/{file}_mordred/{file}_mordred_{i_batch:d}.csv', index=False, float_format='%.4f')

# preprocess labeled dataset
get_labeled_mordred('labeled')

# preprocess hypothetical unlabeled dataset
file = 'unlabeled'
os.mkdir(f'data/{file}_mordred')

for i in range(100):
    get_unlabeled_mordred(file, i)

desc = []
df = pd.read_csv('data/labeled_mordred.csv')
for col in list(df)[3:]:
    if pd.to_numeric(df[col], errors='coerce').notnull().all():
        if np.std(df[col].to_numpy()) > 0:
            desc.append(col)

for i in range(100):
    df = pd.read_csv(f'data/{file}_mordred/{file}_mordred_{i:d}.csv')
    for col in desc:
        if not pd.to_numeric(df[col], errors='coerce').notnull().all():
            desc.remove(col)

df = pd.read_csv('data/labeled_mordred.csv')
df1 = df[['acid', 'epoxide', 'tg'] + desc]
df1.to_csv('data/labeled_mordred.csv', index=False, float_format='%.4f')
np.random.seed(0)
idx = np.random.permutation(len(df1))
train_idx = idx[:int(0.9*len(df1))]
test_idx = idx[int(0.9*len(df1)):]
df1.iloc[train_idx, :].to_csv('data/labeled_mordred_train.csv', index=False, float_format='%.4f')
df1.iloc[test_idx, :].to_csv('data/labeled_mordred_test.csv', index=False, float_format='%.4f')

d = {}
for i in range(100):
    df = pd.read_csv(f'data/{file}_mordred/{file}_mordred_{i:d}.csv')
    for col in ['acid', 'epoxide'] + desc:
        if col in d:
            d[col].extend(df[col].to_list())
        else:
            d[col] = df[col].to_list()

pd.DataFrame(d).to_csv(f'data/{file}_mordred.csv', index=False, float_format='%.4f')
shutil.rmtree(f'data/{file}_mordred')

# preprocess synthesizable unlabeled dataset
file = 'unlabeled_synthesis'
os.mkdir(f'data/{file}_mordred')

get_unlabeled_mordred(file, 0)

desc = []
df = pd.read_csv('data/labeled_mordred.csv')
for col in list(df)[3:]:
    if pd.to_numeric(df[col], errors='coerce').notnull().all():
        if np.std(df[col].to_numpy()) > 0:
            desc.append(col)

df = pd.read_csv(f'data/{file}_mordred/{file}_mordred_0.csv')
for col in desc:
    if not pd.to_numeric(df[col], errors='coerce').notnull().all():
        desc.remove(col)

d = {}
df = pd.read_csv(f'data/{file}_mordred/{file}_mordred_0.csv')
for col in ['acid', 'epoxide'] + desc:
    if col in d:
        d[col].extend(df[col].to_list())
    else:
        d[col] = df[col].to_list()

pd.DataFrame(d).to_csv(f'data/{file}_mordred.csv', index=False, float_format='%.4f')
shutil.rmtree(f'data/{file}_mordred')
