import torch, sys
from torch.utils.data import DataLoader
from transpolymer_dataset import Unlabeled_Dataset
from transpolymer import DownstreamRegression
from transpolymer_tokenizer import PolymerSmilesTokenizer
import pandas as pd
import numpy as np


df_train = pd.read_csv('data/labeled_train.csv')
y_train = df_train['tg'].to_numpy()
y_mean = np.mean(y_train)
y_std = np.std(y_train)

data = 'unlabeled'
df = pd.read_csv(f'data/{data}.csv')

tokenizer = PolymerSmilesTokenizer.from_pretrained('roberta-base', max_len=411)

tg_pred_all = []
for i in range(1, 6):

    model = DownstreamRegression(0.1, 0.1, 0.1).to('cuda')
    model = model.double()
    model.load_state_dict(torch.load(f'results/transpolymer/model{i}.pt', weights_only=True))
    model.eval()

    tg_pred = []
    for j in range(100):
        print(j)
        sys.stdout.flush()
        dataset = Unlabeled_Dataset(df.loc[int(j * 10000):int((j + 1) * 10000 - 1), :], tokenizer, 411)
        dataloader = DataLoader(dataset, batch_size=32)
        for batch in dataloader:
            input_ids = batch["input_ids"].to('cuda')
            attention_mask = batch["attention_mask"].to('cuda')
            y_pred = model(input_ids, attention_mask).float()
            if len(y_pred) == 1:
                tg_pred.append((y_pred * y_std + y_mean).item())
            else:
                tg_pred += (y_pred * y_std + y_mean).tolist()
    tg_pred_all.append(np.array(tg_pred).squeeze())
    print(len(np.array(tg_pred).squeeze()))

acid = df['acid'].to_list()
epoxide = df['epoxide'].to_list()

df = pd.DataFrame({'acid': acid, 'epoxide': epoxide,
                   'tg_pred1': tg_pred_all[0], 
                   'tg_pred2': tg_pred_all[1],
                   'tg_pred3': tg_pred_all[2],
                   'tg_pred4': tg_pred_all[3],
                   'tg_pred5': tg_pred_all[4]})
df.to_csv(f'results/transpolymer/{data}_pred.csv', index=False, float_format='%.4f')
            