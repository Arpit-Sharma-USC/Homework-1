import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df_train=pd.read_csv("F:/INF552/hw3/aps_failure_training_set_backup.csv")#skiprows=20,index_col=21)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv")

imp=Imputer(strategy="mean")

df_train.replace('na',0,inplace=True)
df_test.replace('na',0,inplace=True)

my_l=df_train.columns.values.tolist()
# my_l = my_l[:-1]

my_list_arr=np.array(my_l)
df_train_temp=df_train.astype(float)

CV=df_train_temp.std()/df_train_temp.mean()
CV_= pd.DataFrame({'label':CV.index, 'CV':CV.values})
CV_=CV_.sort_values(by='CV', ascending=False, na_position='first')
print(CV_.values[:,:])
print("Top 13 features:")

my_arr=np.array(CV_.values[:13,1:2])
matching_array=[]

for i in range(0,13):
    matching_array.append(my_arr[i][0])
print(matching_array)

df_for_plot=pd.DataFrame()
for i in range(0,13):
    df_for_plot[matching_array[i]]=df_train_temp[matching_array[i]]
print(df_for_plot)
# plt.figure()
# df_for_plot.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
#
# df_for_plot.plot.box()
# sns.pairplot(df_for_plot)
# scatter_matrix(df_for_plot)

# plt.show()
# plt.savefig("F:/INF552/hw3/test.png")