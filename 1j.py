import xgboost as xgb
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np

df_train=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")#skiprows=20,index_col=21)
df_train.replace('na',0,inplace=True)
df_train.replace('?',0,inplace=True)

X_train=df_train.values[:,1:171]
Y_train=df_train.values[:,:1]

optimized_GBM = GridSearchCV(cv=5,
       estimator=xgb.XGBRegressor(),
       param_grid={'reg_alpha': np.linspace(np.float_power(10, -4), np.float_power(10, 1), 20)},
        refit=True, scoring='neg_mean_squared_error', verbose=1)
# Optimize for accuracy since that is the metric used in the Adult Data Set notation
optimized_GBM.fit(X_train, Y_train)

print(optimized_GBM.grid_scores_)
