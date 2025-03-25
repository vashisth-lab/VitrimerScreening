import pandas as pd
import numpy as np


file = 'unlabeled_synthesis_pred'
models = ['xgb_fp', 'xgb_mordred', 'gnn', 'transpolymer']
tg_pred = []
for model in models:
    df = pd.read_csv(f'results/{model}/{file}.csv')
    for i in range(1, 6):
        tg_pred.append(df[f'tg_pred{i}'].to_list())
        print(df['acid'][:5])

tg_pred = np.array(tg_pred)
tg_pred = np.mean(tg_pred, axis=0)
df_new = pd.DataFrame({'acid': df['acid'].to_list(),
                       'epoxide': df['epoxide'].to_list(),
                       'tg_pred': tg_pred})
df_sorted = df_new.sort_values('tg_pred')
df_sorted.to_csv(f'results/ensemble/{file}.csv', index=False, float_format='%.4f')
