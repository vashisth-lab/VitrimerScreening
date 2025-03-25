import pandas as pd
import numpy as np
import sys, shap, torch, json, itertools
from ffnn import FFNN


representation = sys.argv[1]
i = int(sys.argv[2])

df_train = pd.read_csv('data/labeled_%s_train.csv' % (representation))
X_train = df_train.iloc[:, 3:].to_numpy()
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std
X_train = torch.tensor(X_train, dtype=torch.float)

with open('results/ffnn_%s/hyperparameters.txt' % (representation), 'r') as f:
    d = json.load(f)
model = FFNN(X_train.shape[1], d['hidden layer dimension'], d['number of hidden layers'])
model.load_state_dict(torch.load('results/ffnn_%s/model%d.pt' % (representation, i), weights_only=True, map_location=torch.device('cpu')))
model.eval()

explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_train)

df = pd.DataFrame(data=shap_values.squeeze(), columns=df_train.columns[3:])
df.to_csv('results/ffnn_%s/shap%d.csv' % (representation, i), float_format='%.4f', index=False)
