import numpy as np
import pandas as pd


np.random.seed(0)
df = pd.read_csv('data/labeled.csv')
idx = np.random.permutation(len(df))
train_idx = idx[:int(0.9*len(df))]
test_idx = idx[int(0.9*len(df)):]
df.iloc[train_idx, :].to_csv('data/labeled_train.csv', index=False)
df.iloc[test_idx, :].to_csv('data/labeled_test.csv', index=False)
