import torch, sys, math, os, json, copy, random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class FFNN(nn.Module):
    def __init__(self, d_init, d_hidden, n_hidden):
        super().__init__()
        layers = [nn.Linear(d_init, d_hidden)]
        for _ in range(n_hidden):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(d_hidden, 1))
        self.ffnn = nn.Sequential(*layers)
        
    def forward(self, X):
        return self.ffnn(X)


def train(bs, hd, nh, lr, wd, X_train, X_test, y_train, y_test, y_mean, y_std, best=False):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)

    kf = KFold(n_splits=5)
    min_model_all = []
    train_loss_all = []
    val_loss_all = []
    val_RMSE = 0

    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        model = FFNN(X_train.shape[1], hd, nh).to(torch.device('cuda'))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_train[train_idx], y_train[train_idx]), batch_size=bs, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_train[val_idx], y_train[val_idx]), batch_size=bs)

        min_val_loss = math.inf
        min_model = model.state_dict()
        train_loss = []
        val_loss = []
        
        for _ in range(100):
            model.train()
            epoch_train_loss = 0
            for batch in train_loader:
                X, y_true = batch
                optimizer.zero_grad()
                y_pred = model(X).view(-1)
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
                    X, y_true = batch
                    y_pred = model(X).view(-1)
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
                X, y_true = batch
                y_pred = model(X).view(-1)
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
                      'train_loss5': train_loss_all[4], 'val_loss5': val_loss_all[4]}).to_csv('results/ffnn_%s/loss.txt' % (representation), index=False, float_format='%.4f')

        for i in range(5):
            torch.save(min_model_all[i], 'results/ffnn_%s/model%d.pt' % (representation, i + 1))

        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=bs)

        R2 = []
        MAE = []
        MAPE = []
        RMSE = []
        tg_pred_all = []

        for i in range(5):
            model = FFNN(X_train.shape[1], hd, nh).to(torch.device('cuda'))
            model.load_state_dict(min_model_all[i], strict=True)
            model.eval()
        
            tg_true = []
            tg_pred = []
            with torch.no_grad():
                for batch in test_loader:
                    X, y_true = batch
                    y_pred = model(X).view(-1)
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
              'pred5': tg_pred_all[4]}).to_csv('results/ffnn_%s/pred.txt' % (representation), index=False, float_format='%.4f')
        pd.DataFrame({'R2': R2, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}).to_csv('results/ffnn_%s/metrics.txt' % (representation), index=False, float_format='%.4f')
    return val_RMSE / 5


if __name__ == "__main__":
    representation = sys.argv[1]

    df_train = pd.read_csv('data/labeled_%s_train.csv' % (representation))
    X_train = df_train.iloc[:, 3:].to_numpy()
    y_train = df_train['tg'].to_numpy()

    df_test = pd.read_csv('data/labeled_%s_test.csv' % (representation))
    X_test = df_test.iloc[:, 3:].to_numpy()
    y_test = df_test['tg'].to_numpy()

    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    X_train = torch.tensor(X_train, dtype=torch.float, device=torch.device('cuda'))
    X_test = torch.tensor(X_test, dtype=torch.float, device=torch.device('cuda'))
    y_train = torch.tensor(y_train, dtype=torch.float, device=torch.device('cuda'))
    y_test = torch.tensor(y_test, dtype=torch.float, device=torch.device('cuda'))

    param = []
    rmse_all = []
    for i in range(100):
        random.seed(i)
        bs = random.choice([4, 8, 16, 32, 64, 128])
        hd = random.choice([4, 8, 16, 32, 64, 128])
        nh = random.choice([1, 2, 3])
        lr = random.uniform(1e-4, 1e-2)
        wd = random.uniform(1e-4, 1e-2)
        RMSE = train(bs, hd, nh, lr, wd, X_train, X_test, y_train, y_test, y_mean, y_std)
        print(bs, hd, nh, lr, wd, RMSE)
        sys.stdout.flush()

        param.append({'batch size': bs,
                    'hidden layer dimension': hd,
                    'number of hidden layers': nh,
                    'learning rate': lr,
                    'weight decay': wd})
        rmse_all.append(RMSE)

    idx_min = np.argmin(np.array(rmse_all))
    print('Best RMSE: %.4f' % (rmse_all[idx_min]))

    if not os.path.exists('results/ffnn_%s' % (representation)):
        os.mkdir('results/ffnn_%s' % (representation))

    train(param[idx_min]['batch size'],
        param[idx_min]['hidden layer dimension'],
        param[idx_min]['number of hidden layers'],
        param[idx_min]['learning rate'],
        param[idx_min]['weight decay'],
        X_train, X_test, y_train, y_test, y_mean, y_std, True)

    with open('results/ffnn_%s/hyperparameters.txt' % (representation), 'w') as file:
        file.write(json.dumps(param[idx_min]))
