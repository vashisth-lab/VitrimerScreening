from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import numpy as np
import pandas as pd
from mol2vec.features import mol2alt_sentence
from gensim.models import word2vec
import sys, os, shutil


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


def get_labeled_mol2vec(file):
    # mol2vec with 300 dimensions

    model_path = 'data/model_300dim.pkl'
    model = word2vec.Word2Vec.load(model_path)

    acid = pd.read_csv(f'data/{file}.csv')['acid'].to_list()
    epoxide = pd.read_csv(f'data/{file}.csv')['epoxide'].to_list()
    tg = pd.read_csv(f'data/{file}.csv')['tg'].to_list()
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    sentence = [mol2alt_sentence(m, radius=2) for m in mols]
    vec = [np.mean([model.wv[word] for word in s if word in model.wv], axis=0) for s in sentence]
    vec_all = np.vstack(vec)
    
    d = {'acid': acid, 'epoxide': epoxide, 'tg': tg}
    for i in range(300):
        d['vec' + str(i)] = vec_all[:, i]
    df = pd.DataFrame(d)
    df.to_csv(f'data/{file}_mol2vec.csv', index=False, float_format='%.4f')


def get_unlabeled_mol2vec(file, i_batch):
    model_path = 'data/model_300dim.pkl'
    model = word2vec.Word2Vec.load(model_path)

    acid = pd.read_csv(f'data/{file}.csv')['acid'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    epoxide = pd.read_csv(f'data/{file}.csv')['epoxide'].to_list()[int(i_batch * 10000): int(i_batch * 10000 + 10000)]
    smiles = [vitrimerize(acid[i], epoxide[i]) for i in range(len(acid))]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    sentence = [mol2alt_sentence(m, radius=2) for m in mols]
    vec = [np.mean([model.wv[word] for word in s if word in model.wv], axis=0) for s in sentence]
    vec_all = np.vstack(vec)
    
    d = {'acid': acid, 'epoxide': epoxide}
    for i in range(300):
        d['vec' + str(i)] = vec_all[:, i]
    df = pd.DataFrame(d)
    df.to_csv(f'data/{file}_mol2vec/{file}_mol2vec_{i_batch:d}.csv', index=False, float_format='%.4f')


# preprocess labeled dataset
get_labeled_mol2vec('labeled')

np.random.seed(0)
df = pd.read_csv('data/labeled_mol2vec.csv')
idx = np.random.permutation(len(df))
train_idx = idx[:int(0.9*len(df))]
test_idx = idx[int(0.9*len(df)):]
df.iloc[train_idx, :].to_csv('data/labeled_mol2vec_train.csv', index=False)
df.iloc[test_idx, :].to_csv('data/labeled_mol2vec_test.csv', index=False)

# preprocess hypothetical unlabeled dataset
file = 'unlabeled'
os.mkdir(f'data/{file}_mol2vec')

for i in range(100):
    get_unlabeled_mol2vec(file, i)

d = {}
for i in range(100):
    df = pd.read_csv(f'data/{file}_mol2vec/{file}_mol2vec_{i:d}.csv')
    for col in df:
        if col in d:
            d[col].extend(df[col].to_list())
        else:
            d[col] = df[col].to_list()

pd.DataFrame(d).to_csv(f'data/{file}_mol2vec.csv', index=False, float_format='%.4f')
shutil.rmtree(f'data/{file}_mol2vec')

# preprocess synthesizable unlabeled dataset
file = 'unlabeled_synthesis'
os.mkdir(f'data/{file}_mol2vec')

get_unlabeled_mol2vec(file, 0)

d = {}
df = pd.read_csv(f'data/{file}_mol2vec/{file}_mol2vec_0.csv')
for col in df:
    if col in d:
        d[col].extend(df[col].to_list())
    else:
        d[col] = df[col].to_list()

pd.DataFrame(d).to_csv(f'data/{file}_mol2vec.csv', index=False, float_format='%.4f')
shutil.rmtree(f'data/{file}_mol2vec')
