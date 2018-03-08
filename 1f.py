import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")
df=df.replace('?',0)

X_train=df.values[:1495,:126]
Y_train=df.values[:1495,126:]

X_test=df.values[1495:,:126]
Y_test=df.values[1495:,126:]

model=linear_model.LinearRegression()
model.fit(X_train,Y_train)

preds=model.predict(X_test)
error=mean_squared_error(Y_test,preds)

print("Test Error:",error)
