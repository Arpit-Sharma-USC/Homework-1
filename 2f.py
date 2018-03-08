##################  model trees with test set SMOTE##############################
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.utils import resample

import proc as prc
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc

df_train=pd.read_csv("F:/INF552/hw3/aps_failure_training_set.csv")#skiprows=20,index_col=21)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv")

df_train.replace('na',0,inplace=True)
df_test.replace('na',0,inplace=True)

sm = SMOTE(random_state=42)

X_train_u=df_train.values[:,1:171]
Y_train_u=df_train.values[:,:1]

X_train, Y_train = sm.fit_sample(X_train_u, Y_train_u.ravel())

X_test_u=df_test.values[:,1:171]
Y_test_u=df_test.values[:,:1]

sm_test=SMOTE(random_state=42)
X_test, Y_test = sm.fit_sample(X_test_u, Y_test_u.ravel())

#
# model=RandomForestClassifier(max_depth=5)
# model.fit(X_train,Y_train.ravel())
# enc=OneHotEncoder()
# enc.fit(model.apply(X_train))
# temp_model=model.fit(enc.transform(model.apply(X_test)),Y_test.ravel())
# y_pred_rf = model.predict_proba(enc.transform(model.apply(X_test)))[:, 1]

# model_prob = model.predict_proba(X_test)
# score=log_loss(Y_test,model_prob)
# score_mean=mean_squared_error(Y_test,model.predict(X_test))
# print("Score:",score)
# print("MSE:",score_mean)
# print("model_prob",model_prob)

# model_prob=model_prob.reshape(1,-1)
#
rf = RandomForestClassifier(max_depth=3)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegressionCV(cv=5)
rf.fit(X_train, Y_train.ravel())
rf_enc.fit(rf.apply(X_train))
model_used=rf_lm.fit(rf_enc.transform(rf.apply(X_test)), Y_test.ravel())
preds=rf.predict(X_test)
print(type(model_used))

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_test, y_pred_rf_lm,pos_label='pos')
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)

plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
plt.title('ROC curve-Test Set (With Imbalance removal using SMOTE), AUC: '+str(roc_auc))
plt.xlabel('False positive rate')
plt.plot([0, 1], [0, 1], 'k--')
# plt.legend('AUC:'str(round(roc_auc)))
plt.ylabel('True positive rate')
print(roc_auc)
plt.show()
confusion_matrx=confusion_matrix(Y_test,preds)
print(confusion_matrx)
sns.heatmap(confusion_matrx,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
plt.show()
missclassifications=0
for i in range(0,2):
    for j in range(0,2):
        if i!=j:
            missclassifications+=confusion_matrx[i][j]
print(missclassifications)
print("missclassification rate:", missclassifications/31250)


#