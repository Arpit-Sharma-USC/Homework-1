import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import numpy as np

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")
df=df.replace('?',0)

X_train=df.values[:1495,:126]
Y_train=df.values[:1495,126:]

X_test=df.values[1495:,:126]
Y_test=df.values[1495:,126:]
arr=np.linspace(0.01,100.1,100)
# print(arr)
model=RidgeCV(alphas=arr,cv=5)#[53.3])#lambda
model.fit(X_train,Y_train)

preds=model.predict(X_test)
error=mean_squared_error(Y_test,preds)
print("Score:",model.score(X_test,Y_test))
print("Test Error:",error)
print("penalty",model.alpha_)
