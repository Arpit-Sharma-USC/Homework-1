from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from operator import itemgetter
from collections import OrderedDict



import pandas as pd

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")
df.replace('?',0,inplace=True)
pca = PCA()

X_train=df.values[:1495,:126]
Y_train=df.values[:1495,126:]

X_test=df.values[1495:,:126]
Y_test=df.values[1495:,126:]
regr = LinearRegression()


pca2 = PCA()

# Split into training and test sets

# Scale the data
X_reduced_train = pca2.fit_transform(scale(X_train))
n = len(X_reduced_train)

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold( n_splits=5, shuffle=True, random_state=1)

mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1*model_selection.cross_val_score(regr, np.ones((n,1)), Y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
mse.append(score)
my_dictionary={}
# Calculate MSE using CV for the 19 principle components, adding one component at the time.
for i in np.arange(1, 126):
    score = -1*model_selection.cross_val_score(regr, X_reduced_train[:,:i], Y_train.ravel(), cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    my_dictionary[i]=score
temp=np.array(mse)

my_dictionary=OrderedDict(sorted(my_dictionary.items(), key=lambda t: t[1]))

print(temp)
print("dictionary",my_dictionary)
plt.plot(temp, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('Salary')
plt.xlim(xmin=-1)
plt.show()
#
M = list(my_dictionary.keys())[0]
print(M)
X_reduced_test = pca2.transform(scale(X_test))[:,:M+1]

# Train regression model on training data
regr = LinearRegression()
regr.fit(X_reduced_train[:,:M+1], Y_train)

# Prediction with test data
pred = regr.predict(X_reduced_test)
error_pcr=mean_squared_error(Y_test, pred)
print("Error:",error_pcr)
print("Accuracy:",regr.score(X_reduced_test,Y_test))