from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import AllChem


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



class Downstream_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_token_len = max_token_len

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        acid = self.dataset.iloc[i, 0]
        epoxide = self.dataset.iloc[i, 1]
        seq = vitrimerize(acid, epoxide)
        prop = self.dataset.iloc[i, 2]

        encoding = self.tokenizer(
            str(seq),
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            prop=prop
        )


