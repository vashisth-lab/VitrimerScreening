from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
import os, shutil


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


def get_labeled_fp(file, n_fp):
    # select n_fp fingerprints that occur in most of the molecules

    acid = pd.read_csv('data/' + file + '.csv')['acid'].to_list()
    epoxide = pd.read_csv('data/' + file + '.csv')['epoxide'].to_list()
    tg = pd.read_csv('data/' + file + '.csv')['tg'].to_list()
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = [AllChem.GetMorganFingerprint(m, radius=3) for m in mols]
    fp_on = [f.GetNonzeroElements() for f in fp]

    fp_unique_list = set()
    for f in fp_on:
        for k in f.keys():
            fp_unique_list.add(k)
    fp_unique_list = list(fp_unique_list)

    fp_freq = {}
    for f1 in fp_unique_list:
        for f2 in fp_on:
            if f1 in f2.keys():
                fp_freq[f1] = fp_freq.get(f1, 0) + 1
    
    fp_freq_sorted = dict(sorted(fp_freq.items(), key=lambda item: item[1]))
    fp_final = list(fp_freq_sorted.keys())[-n_fp:]

    n_mol = len(smiles)
    fp_freq_final = np.zeros((n_mol, n_fp), dtype=int)
    for i in range(n_mol):
        for j in range(n_fp):
            if fp_final[j] in fp_on[i].keys():
                fp_freq_final[i, j] = fp_on[i][fp_final[j]]
    
    d = {'acid': acid, 'epoxide': epoxide, 'tg': tg}
    for i, f in enumerate(fp_final):
        if np.std(fp_freq_final[:, i]) > 0:
            d[f] = fp_freq_final[:, i]
    df = pd.DataFrame(d)
    df.to_csv('data/' + file + '_fp.csv', index=False)

    os.mkdir('data/fp_img')
    for col in df:
        if col != 'acid' and col != 'epoxide' and col != 'tg':
            occur = df[col].to_numpy()
            idx = np.argwhere(occur > 0)[0][0]
            bit = int(col)
            mol = Chem.MolFromSmiles(smiles[idx])
            bi = {}
            fp = AllChem.GetMorganFingerprint(mol, radius=3, bitInfo=bi)
            p = Draw.DrawMorganBit(mol, bit, bi)
            with open ('data/fp_img/%s.svg' % (col), 'w') as f:
                f.write(p)


def get_unlabeled_fp(file, fp_list, i_batch):
    n_fp = len(fp_list)

    acid = pd.read_csv(f'data/{file}.csv')['acid'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    epoxide = pd.read_csv(f'data/{file}.csv')['epoxide'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = [AllChem.GetMorganFingerprint(m, radius=3) for m in mols]
    fp_on = [f.GetNonzeroElements() for f in fp]

    n_mol = len(smiles)
    fp_freq_final = np.zeros((n_mol, n_fp), dtype=int)
    for i in range(n_mol):
        for j in range(n_fp):
            if fp_list[j] in fp_on[i].keys():
                fp_freq_final[i, j] = fp_on[i][fp_list[j]]
    
    d = {'acid': acid, 'epoxide': epoxide}
    for i, f in enumerate(fp_list):
        d[f] = fp_freq_final[:, i]
    df = pd.DataFrame(d)
    df.to_csv(f'data/{file}_fp/{file}_fp_{i_batch:d}.csv', index=False)


# preprocess labeled dataset
get_labeled_fp('labeled', 200)

np.random.seed(0)
df = pd.read_csv('data/labeled_fp.csv')
idx = np.random.permutation(len(df))
train_idx = idx[:int(0.9*len(df))]
test_idx = idx[int(0.9*len(df)):]
df.iloc[train_idx, :].to_csv('data/labeled_fp_train.csv', index=False)
df.iloc[test_idx, :].to_csv('data/labeled_fp_test.csv', index=False)

# preprocess hypothetical unlabeled dataset
file = 'unlabeled'
os.mkdir(f'data/{file}_fp')

fp_list = list(pd.read_csv('data/labeled_fp.csv'))[3:]
fp_list = [int(f) for f in fp_list]

for i in range(100):
    get_unlabeled_fp(file, fp_list, i)

d = {}
for i in range(100):
    df = pd.read_csv(f'data/{file}_fp/{file}_fp_{i:d}.csv')
    for col in df:
        if col in d:
            d[col].extend(df[col].to_list())
        else:
            d[col] = df[col].to_list()

pd.DataFrame(d).to_csv(f'data/{file}_fp.csv', index=False)
shutil.rmtree(f'data/{file}_fp')

# preprocess synthesizable unlabeled dataset
file = 'unlabeled_synthesis'
os.mkdir(f'data/{file}_fp')

fp_list = list(pd.read_csv('data/labeled_fp.csv'))[3:]
fp_list = [int(f) for f in fp_list]

get_unlabeled_fp(file, fp_list, 0)

d = {}
df = pd.read_csv(f'data/{file}_fp/{file}_fp_0.csv')
for col in df:
    if col in d:
        d[col].extend(df[col].to_list())
    else:
        d[col] = df[col].to_list()

pd.DataFrame(d).to_csv(f'data/{file}_fp.csv', index=False)
shutil.rmtree(f'data/{file}_fp')
