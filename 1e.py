import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")

df=df.replace('?',0)

my_l=df.columns.values.tolist()

my_list_arr=np.array(my_l)

CV=df.std()/df.mean()
CV_= pd.DataFrame({'label':CV.index, 'CV':CV.values})
CV_=CV_.sort_values(by='CV', ascending=False, na_position='first')

print("Top 11 features:")

my_arr=np.array(CV_.values[:11,1:2])
matching_array=[]

for i in range(0,11):
    matching_array.append(my_arr[i][0])
print(matching_array)

df_for_plot=pd.DataFrame()
for i in range(0,11):
    df_for_plot[matching_array[i]]=df[matching_array[i]]
print(df_for_plot)

df_for_plot.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
df_for_plot.plot.box()
scatter_matrix(df_for_plot)
plt.show()

