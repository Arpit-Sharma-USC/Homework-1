import numpy as np
import pandas as pd

df_train=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_training_set.csv",skiprows=20)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv",skiprows=20)

df_train.replace('na',0,inplace=True)
df_test.replace('na',0,inplace=True)

X_train=df_train.values[:,1:]
Y_train=df_train.values[:,:1]


X_test=df_test.values[:,1:]
Y_test=df_test.values[:,:1]
#
# arr=np.array(df_train.iloc[:,1:2])
# print(type(arr))
# print(arr)
i=2
# for i in range(1,171):
# m=df_train.iloc[:,3:4].mean()
# s=df_train.iloc[:,3:4].var()
# temp=s/m
# print(m)
# print(s)
print(type(df_train))
# print(df_train)
print(df_train.iloc[:,[2]].mean().astype(float))

# s=df_train.iloc[:,3:4].var()
# temp=s/m
# print(m)
# print(s)
# print("cal-cov for feature:",i,":",temp)