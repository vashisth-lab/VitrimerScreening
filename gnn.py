import torch, random, os, math, copy, sys, json
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class GraphDataset(InMemoryDataset):
    def __init__(self, file):
        super().__init__()
        self.data, self.slices = torch.load('data/%s.pt' % (file))


class GCN(nn.Module):
    def __init__(self, n_convolutions, convolutions_dim, n_hidden_layers, hidden_layers_dim):
        super(GCN, self).__init__()
                
        self.n_convolutions = n_convolutions
        self.n_hidden_layers = n_hidden_layers

        self.layers = nn.ModuleList()

        for i in range(n_convolutions):
            self.layers.append(GCNConv(-1 if i == 0 else convolutions_dim, convolutions_dim))
            self.layers.append(BatchNorm1d(convolutions_dim))

        for i in range(n_hidden_layers):
            self.layers.append(Linear(convolutions_dim if i == 0 else hidden_layers_dim, hidden_layers_dim))
            self.layers.append(BatchNorm1d(hidden_layers_dim))

        self.layers.append(Linear(hidden_layers_dim, 1))
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        n_conv_batch = 2 * self.n_convolutions
        n_hidden_batch = 2 * self.n_hidden_layers

        for idx, layer in enumerate(self.layers[:n_conv_batch]):
            if idx % 2 == 0:
                x = F.relu(layer(x, edge_index))
            else:
                x = layer(x)

        x = global_add_pool(x, data.batch)

        for idx, layer in enumerate(self.layers[n_conv_batch:]):
            if idx % 2 == 0 and idx != n_hidden_batch:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        return x


def train(bs, cd, nc, hd, nh, lr, wd, train_set, test_set, y_mean, y_std, best=False):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)

    kf = KFold(n_splits=5)
    min_model_all = []
    train_loss_all = []
    val_loss_all = []
    val_RMSE = 0

    for i, (train_idx, val_idx) in enumerate(kf.split(train_set)):

        model = GCN(nc, cd, nh, hd).to(torch.device('cuda'))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_set[list(train_idx)], batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(train_set[list(val_idx)], batch_size=bs)

        min_val_loss = math.inf
        min_model = model.state_dict()
        train_loss = []
        val_loss = []
        
        for _ in range(100):
            model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                y_true = (batch.y - y_mean) / y_std
                optimizer.zero_grad()
                y_pred = model(batch).view(-1)
                loss = criterion(y_true, y_pred)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            epoch_train_loss /= len(train_loader)
            train_loss.append(epoch_train_loss)

            epoch_val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    y_true = (batch.y - y_mean) / y_std
                    y_pred = model(batch).view(-1)
                    loss = criterion(y_true, y_pred)
                    epoch_val_loss += loss.item()
            epoch_val_loss /= len(val_loader)
            val_loss.append(epoch_val_loss)

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                min_model = copy.deepcopy(model.state_dict())
            
        if best:
            min_model_all.append(min_model)
            train_loss_all.append(train_loss)
            val_loss_all.append(val_loss)

        tg_true = []
        tg_pred = []
        model.load_state_dict(min_model, strict=True)
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                y_true = (batch.y - y_mean) / y_std
                y_pred = model(batch).view(-1)
                if len(y_pred) == 1:
                    tg_true.append((y_true * y_std + y_mean).item())
                    tg_pred.append((y_pred * y_std + y_mean).item())
                else:
                    tg_true += (y_true * y_std + y_mean).tolist()
                    tg_pred += (y_pred * y_std + y_mean).tolist()
        val_RMSE += mean_squared_error(tg_true, tg_pred, squared=False)

    if best:
        pd.DataFrame({'train_loss1': train_loss_all[0], 'val_loss1': val_loss_all[0], 
                      'train_loss2': train_loss_all[1], 'val_loss2': val_loss_all[1], 
                      'train_loss3': train_loss_all[2], 'val_loss3': val_loss_all[2], 
                      'train_loss4': train_loss_all[3], 'val_loss4': val_loss_all[3], 
                      'train_loss5': train_loss_all[4], 'val_loss5': val_loss_all[4]}).to_csv('results/gnn/loss.txt', index=False, float_format='%.4f')
        
        for i in range(5):
            torch.save(min_model_all[i], 'results/gnn/model%d.pt' % (i + 1))

        test_loader = DataLoader(test_set, batch_size=bs)

        R2 = []
        MAE = []
        MAPE = []
        RMSE = []
        tg_pred_all = []

        for i in range(5):
            model = GCN(nc, cd, nh, hd).to(torch.device('cuda'))
            model.load_state_dict(min_model_all[i], strict=True)
            model.eval()

            tg_true = []
            tg_pred = []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    y_true = (batch.y - y_mean) / y_std
                    y_pred = model(batch).view(-1)
                    if len(y_pred) == 1:
                        tg_true.append((y_true * y_std + y_mean).item())
                        tg_pred.append((y_pred * y_std + y_mean).item())
                    else:
                        tg_true += (y_true * y_std + y_mean).tolist()
                        tg_pred += (y_pred * y_std + y_mean).tolist()
            
            R2.append(r2_score(tg_true, tg_pred))
            MAE.append(mean_absolute_error(tg_true, tg_pred))
            MAPE.append(mean_absolute_percentage_error(tg_true, tg_pred))
            RMSE.append(mean_squared_error(tg_true, tg_pred, squared=False))
            tg_pred_all.append(tg_pred)

        pd.DataFrame({'true': tg_true,
              'pred1': tg_pred_all[0],
              'pred2': tg_pred_all[1], 
              'pred3': tg_pred_all[2],
              'pred4': tg_pred_all[3],
              'pred5': tg_pred_all[4]}).to_csv('results/gnn/pred.txt', index=False, float_format='%.4f')
        pd.DataFrame({'R2': R2, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}).to_csv('results/gnn/metrics.txt', index=False, float_format='%.4f')
    return val_RMSE / 5


if __name__ == '__main__':
    df_train = pd.read_csv('data/labeled_train.csv')
    y_train = df_train['tg'].to_numpy()

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    train_set = GraphDataset('labeled_train_graph').cuda()
    test_set = GraphDataset('labeled_test_graph').cuda()

    param = []
    rmse_all = []
    for i in range(100):
        random.seed(i)
        bs = random.choice([4, 8, 16, 32, 64, 128])
        cd = random.choice([4, 8, 16, 32, 64, 128])
        nc = random.choice([1, 2, 3])
        hd = random.choice([4, 8, 16, 32, 64, 128])
        nh = random.choice([1, 2, 3])
        lr = random.uniform(1e-4, 1e-2)
        wd = random.uniform(1e-4, 1e-2)
        RMSE = train(bs, cd, nc, hd, nh, lr, wd, train_set, test_set, y_mean, y_std)
        print(bs, cd, nc, hd, nh, lr, wd, RMSE)
        sys.stdout.flush()

        param.append({'batch size': bs,
                    'convolutional layer dimension': cd,
                    'number of convolutional layers': nc,
                    'hidden layer dimension': hd,
                    'number of hidden layers': nh,
                    'learning rate': lr,
                    'weight decay': wd})
        rmse_all.append(RMSE)

    idx_min = np.argmin(np.array(rmse_all))
    print('Best RMSE: %.4f' % (rmse_all[idx_min]))

    if not os.path.exists('results/gnn'):
        os.mkdir('results/gnn')

    train(param[idx_min]['batch size'],
        param[idx_min]['convolutional layer dimension'],
        param[idx_min]['number of convolutional layers'],
        param[idx_min]['hidden layer dimension'],
        param[idx_min]['number of hidden layers'],
        param[idx_min]['learning rate'],
        param[idx_min]['weight decay'],
        train_set, test_set, y_mean, y_std, True)

    with open('results/gnn/hyperparameters.txt', 'w') as file:
        file.write(json.dumps(param[idx_min]))
