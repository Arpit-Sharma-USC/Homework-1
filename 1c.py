import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")
df=df.replace('?',0)

correlations=df.corr()
print(correlations)
plt.figure(figsize=(100,80))
sns.heatmap(correlations,cmap="YlGnBu")
plt.show()

