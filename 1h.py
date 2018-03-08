import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import numpy as np

df=pd.read_csv("C:/Users/Shanu/PycharmProjects/Crime-data/communities.csv")
df=df.replace('?',0)

my_l=df.columns.values.tolist()
my_l = my_l[:-1]

my_list_arr=np.array(my_l)

X_train=df.values[:1495,:126]
Y_train=df.values[:1495,126:]

X_test=df.values[1495:,:126]
Y_test=df.values[1495:,126:]
arr=np.linspace(0.001,100.1,3000)
print(arr)

model=LassoCV(alphas=arr,cv=5)
model.fit(X_train,Y_train)

preds=model.predict(X_test)
error=mean_squared_error(Y_test,preds)

print("Score:",model.score(X_test,Y_test))
print("Test Error:",error)

print("penalty",model.alpha_)

final_features=model.coef_
# print("Coef_path:",model.coef_)
# print(final_features.shape)
print(type(final_features))
temp=0
for i in range (0,126):
    if(final_features[i]!=0):
        print(my_list_arr[i],":",final_features[i])
        temp+=1
print(temp)