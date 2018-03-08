import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")

# print(dataset.values[0][1])
df=df.replace('?',0)

my_l=df.columns.values.tolist()
# my_l = my_l[:-1]

my_list_arr=np.array(my_l)

CV=df.std()/df.mean()
CV_= pd.DataFrame({'label':CV.index, 'CV':CV.values})
print("CV values features:")
print(CV)
# CV_=CV_.sort_values(by='CV', ascending=False, na_position='first')

#
# my_arr=np.array(CV_.values[:,:])
# print(CV_.values[:,:])
# matching_array=[]
#
# for i in range(0,11):
#     matching_array.append(my_arr[i][0])
# print(matching_array)
#
# df_for_plot=pd.DataFrame()
# for i in range(0,11):
#     df_for_plot[matching_array[i]]=df[matching_array[i]]
# print(df_for_plot)
#
# df_for_plot.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
#
# df_for_plot.plot.box()
#
# scatter_matrix(df_for_plot)
#
# plt.show()
# #
# #
#
# # temp=np.array(CV)
#
# # print(temp.shape)
# # temp=-np.sort(-temp)
# # # print(my_list_arr)
# # print(temp)
#
# # print(my_list_arr.shape)
# # X=dataset.values[:,:3]
# # X_p=dataset.values[:,3:4]
# # X_pp=dataset.values[:,4:127]
# #
# # Y=dataset.values[:,127:]
# #
# # print(X)
# # print("now Y","\n",Y)
# # try:
# # imp=Imputer(missing_values="NaN", strategy='mean', axis=0, verbose=0, copy=True)
# # imp.fit(X,Y)
# # print(X)
# # # imp.fit(X_p)
# # imp.fit(X_pp)
#
# # print(X)
#
