import pandas as pd
import sys, os, pickle


representation = sys.argv[1]

if not os.path.exists('results/xgb_%s' % (representation)):
     os.mkdir('results/xgb_%s' % (representation))

seed = 0

data = 'unlabeled_synthesis'
df = pd.read_csv(f'data/{data}_{representation}.csv')
acid = df['acid'].to_list()
epoxide = df['epoxide'].to_list()
X = df.iloc[:, 2:].to_numpy()

tg_pred = []
for i in range(1, 6):
     with open(f'results/xgb_{representation}/model{i:d}.pkl', 'rb') as f:
          pipe = pickle.load(f)
          tg_pred.append(pipe.predict(X))

df = pd.DataFrame({'acid': acid, 'epoxide': epoxide,
                   'tg_pred1': tg_pred[0], 
                   'tg_pred2': tg_pred[1],
                   'tg_pred3': tg_pred[2],
                   'tg_pred4': tg_pred[3],
                   'tg_pred5': tg_pred[4]})
df.to_csv(f'results/xgb_{representation}/{data}_pred.csv', index=False, float_format='%.4f')
