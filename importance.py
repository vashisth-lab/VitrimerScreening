import pandas as pd
import sys, pickle, shap, itertools


representation = sys.argv[1]
model = sys.argv[2]
i = int(sys.argv[3])

df_train = pd.read_csv('data/labeled_%s_train.csv' % (representation))
X_train = df_train.iloc[:, 3:].to_numpy()

with open(f'results/{model}_{representation}/model{i:d}.pkl', 'rb') as f:
    pipe = pickle.load(f)

explainer = shap.Explainer(pipe.predict, X_train)
shap_values = explainer.shap_values(X_train)

df = pd.DataFrame(data=shap_values, columns=df_train.columns[3:])
df.to_csv(f'results/{model}_{representation}/shap{i:d}.csv', float_format='%.4f', index=False)
