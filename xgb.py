import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import uniform, loguniform
import sys, json, os, pickle


representation = sys.argv[1]

if not os.path.exists('results/xgb_%s' % (representation)):
     os.mkdir('results/xgb_%s' % (representation))

seed = 0

df_train = pd.read_csv('data/labeled_%s_train.csv' % (representation))
X_train = df_train.iloc[:, 3:].to_numpy()
y_train = df_train['tg'].to_numpy()

df_test = pd.read_csv('data/labeled_%s_test.csv' % (representation))
X_test = df_test.iloc[:, 3:].to_numpy()
y_test = df_test['tg'].to_numpy()

pipe = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor(n_jobs=None))])
params = {'xgb__n_estimators': list(range(10, 300)),
        'xgb__max_depth': list(range(1, 100)),
        'xgb__learning_rate': uniform(0.01, 0.29),
        'xgb__subsample': uniform(0.5, 0.5),
        'xgb__colsample_bytree': uniform(0.5, 0.5),
        'xgb__min_child_weight': uniform(1, 9),
        'xgb__reg_lambda': uniform(0, 10),
        'xgb__reg_alpha': uniform(0, 10)}
rs = RandomizedSearchCV(pipe, param_distributions=params, cv=5, n_iter=100, refit=False, 
                        scoring='neg_root_mean_squared_error', random_state=seed, n_jobs=-1, verbose=3)
rs.fit(X_train, y_train)

with open('results/xgb_%s/hyperparameters.txt' % (representation), 'w') as file:
     file.write(json.dumps(rs.best_params_))

R2 = []
MAE = []
MAPE = []
RMSE = []
y_pred_all = []
kf = KFold(n_splits=5)
for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
     pipe = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor(n_estimators=rs.best_params_['xgb__n_estimators'], 
                                                                         max_depth=rs.best_params_['xgb__max_depth'], 
                                                                         learning_rate=rs.best_params_['xgb__learning_rate'], 
                                                                         subsample=rs.best_params_['xgb__subsample'], 
                                                                         colsample_bytree=rs.best_params_['xgb__colsample_bytree'], 
                                                                         min_child_weight=rs.best_params_['xgb__min_child_weight'], 
                                                                         reg_lambda=rs.best_params_['xgb__reg_lambda'], 
                                                                         reg_alpha=rs.best_params_['xgb__reg_alpha'], 
                                                                         random_state=seed))])
     pipe.fit(X_train[train_idx, :], y_train[train_idx])
     y_pred = pipe.predict(X_test)
     R2.append(r2_score(y_test, y_pred))
     MAE.append(mean_absolute_error(y_test, y_pred))
     MAPE.append(mean_absolute_percentage_error(y_test, y_pred))
     RMSE.append(root_mean_squared_error(y_test, y_pred))
     y_pred_all.append(y_pred)

     with open('results/xgb_%s/model%d.pkl' % (representation, i+1), 'wb') as f:
          pickle.dump(pipe, f)

pd.DataFrame({'true': y_test,
              'pred1': y_pred_all[0],
              'pred2': y_pred_all[1], 
              'pred3': y_pred_all[2],
              'pred4': y_pred_all[3],
              'pred5': y_pred_all[4]}).to_csv('results/xgb_%s/pred.txt' % (representation), index=False, float_format='%.4f')
pd.DataFrame({'R2': R2, 'MAE': MAE, 'MAPE': MAPE, 'RMSE': RMSE}).to_csv('results/xgb_%s/metrics.txt' % (representation), index=False, float_format='%.4f')

