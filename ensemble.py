import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import os


tg_true = pd.read_csv('data/labeled_test.csv')['tg'].to_numpy()
tg_pred = []
for model in ['ffnn', 'lasso', 'rf', 'svr', 'xgb']:
    for repre in ['fp', 'mol2vec', 'mordred', 'rdkit']:
        df = pd.read_csv(f'results/{model}_{repre}/pred.txt')
        for i in range(1, 6):
            tg_pred.append(df[f'pred{i:d}'].to_numpy())

df = pd.read_csv('results/gnn/pred.txt')
for i in range(1, 6):
    tg_pred.append(df[f'pred{i:d}'].to_numpy())

df = pd.read_csv('results/transpolymer/pred.txt')
for i in range(1, 6):
    tg_pred.append(df[f'pred{i:d}'].to_numpy())

tg_pred = np.array(tg_pred)
tg_pred = np.mean(tg_pred, axis=0)

R2 = []
MAE = []
MAPE = []
RMSE = []

R2.append(r2_score(tg_true, tg_pred))
MAE.append(mean_absolute_error(tg_true, tg_pred))
MAPE.append(mean_absolute_percentage_error(tg_true, tg_pred))
RMSE.append(mean_squared_error(tg_true, tg_pred, squared=False))

if not os.path.exists('results/ensemble_all'):
     os.mkdir('results/ensemble_all')

pd.DataFrame({'R2': R2, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}).to_csv('results/ensemble_all/metrics.txt', index=False, float_format='%.4f')
pd.DataFrame({'true': tg_true,
              'pred': tg_pred}).to_csv('results/ensemble_all/pred.txt', index=False, float_format='%.4f')


tg_true = pd.read_csv('data/labeled_test.csv')['tg'].to_numpy()
tg_pred = []
for model in ['xgb_fp', 'xgb_mordred', 'gnn', 'transpolymer']:
    df = pd.read_csv(f'results/{model}/pred.txt')
    for i in range(1, 6):
        tg_pred.append(df[f'pred{i:d}'].to_numpy())

tg_pred = np.array(tg_pred)
tg_pred = np.mean(tg_pred, axis=0)

R2 = []
MAE = []
MAPE = []
RMSE = []

R2.append(r2_score(tg_true, tg_pred))
MAE.append(mean_absolute_error(tg_true, tg_pred))
MAPE.append(mean_absolute_percentage_error(tg_true, tg_pred))
RMSE.append(mean_squared_error(tg_true, tg_pred, squared=False))

path = 'ensemble_wo_transpolymer'
if not os.path.exists(f'results/{path}'):
     os.mkdir(f'results/{path}')

pd.DataFrame({'R2': R2, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}).to_csv(f'results/{path}/metrics.txt', index=False, float_format='%.4f')
pd.DataFrame({'true': tg_true,
              'pred': tg_pred}).to_csv(f'results/{path}/pred.txt', index=False, float_format='%.4f')
