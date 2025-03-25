import torch, json
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from gnn import GraphDataset, GCN


df_train = pd.read_csv('data/labeled_train.csv')
y_train = df_train['tg'].to_numpy()
y_mean = np.mean(y_train)
y_std = np.std(y_train)

data = 'unlabeled_synthesis'

tg_pred_all = []
for i in range(1, 6):

    with open('results/gnn/hyperparameters.txt', 'r') as f:
        d = json.load(f)
    model = GCN(d['number of convolutional layers'],
                d['convolutional layer dimension'],
                d['number of hidden layers'],
                d['hidden layer dimension']).to(torch.device('cuda'))
    model.load_state_dict(torch.load(f'results/gnn/model{i}.pt', weights_only=True))
    model.eval()

    tg_pred = []
    for j in range(1):
        dataset = GraphDataset(f'{data}_graph/{data}_graph_{j}').cuda()
        dataloader = DataLoader(dataset, batch_size=32)
        for batch in dataloader:
            y_pred = model(batch).view(-1)
            if len(y_pred) == 1:
                tg_pred.append((y_pred * y_std + y_mean).item())
            else:
                tg_pred += (y_pred * y_std + y_mean).tolist()
    tg_pred_all.append(tg_pred)

df = pd.read_csv(f'data/{data}.csv')
acid = df['acid'].to_list()
epoxide = df['epoxide'].to_list()

df = pd.DataFrame({'acid': acid, 'epoxide': epoxide,
                   'tg_pred1': tg_pred_all[0], 
                   'tg_pred2': tg_pred_all[1],
                   'tg_pred3': tg_pred_all[2],
                   'tg_pred4': tg_pred_all[3],
                   'tg_pred5': tg_pred_all[4]})
df.to_csv(f'results/gnn/{data}_pred.csv', index=False, float_format='%.4f')
            