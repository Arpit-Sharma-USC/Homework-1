import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer

df_train=pd.read_csv("F:/INF552/hw3/aps_failure_training_set_backup.csv")#skiprows=20,index_col=21)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv")

imp=Imputer(strategy="mean")

df_train.replace('na',0,inplace=True)
df_test.replace('na',0,inplace=True)
df_train_temp=df_train.astype(float)

X=df_train_temp.values[:,1:]


X_train=df_train.values[:,1:170]
Y_train=df_train.values[:,170:]


X_test=df_test.values[:,1:]
Y_test=df_test.values[:,:1]

corr=df_train_temp.corr()

plt.figure(figsize=(80,80))
sns.heatmap(corr)
plt.show()
plt.savefig("F:/INF552/hw3/corr_2b.png")



