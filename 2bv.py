import pandas as pd

df_train=pd.read_csv("F:/INF552/hw3/aps_failure_training_set.csv")#skiprows=20,index_col=21)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv")

df_majority = df_train[df_train.classs == "neg"]
df_minority = df_train[df_train.classs == "pos"]

print("Training set:")

print(df_majority.classs.value_counts())

print(df_minority.classs.value_counts())

df_majority = df_test[df_test.classs == "neg"]
df_minority = df_test[df_test.classs == "pos"]


print("Testing set:")

print(df_majority.classs.value_counts())

print(df_minority.classs.value_counts())


