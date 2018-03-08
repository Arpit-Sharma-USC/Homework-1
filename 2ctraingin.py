############################# simple RF with training set with Imbalance ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import proc as prc
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc

df_train=pd.read_csv("F:/INF552/hw3/aps_failure_training_set.csv")#skiprows=20,index_col=21)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv")


df_train.replace('na',0,inplace=True)
df_test.replace('na',0,inplace=True)
X_train=df_train.values[:,1:171]
Y_train=df_train.values[:,:1]

#
# X_test=df_test.values[:,1:171]
# Y_test=df_test.values[:,:1]

# X_train,X_test,Y_train,Y_test=train_test_split(X_train_prime,Y_train_prime,test_size=0.2, random_state=42)


model=RandomForestClassifier(max_depth=5)
model.fit(X_train,Y_train.ravel())

# model_prob = model.predict_proba(X_test)
# score=log_loss(Y_test,model_prob)
#
# print("Score:",score)
# # print("MSE:",score_mean)
# print("model_prob",model_prob)

#
rf = RandomForestClassifier(max_depth=3,oob_score=True)
rf_enc = OneHotEncoder()
# rf_lm = SGDClassifier()
rf_lm = SVC(probability=True)

rf.fit(X_train, Y_train.ravel())
rf_enc.fit(rf.apply(X_train))
model_used=rf_lm.fit(rf_enc.transform(rf.apply(X_train)), Y_train.ravel())
preds=rf.predict(X_train)
print(type(model_used))


y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_train)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_train, y_pred_rf_lm,pos_label='pos')
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)
print(fpr_rf_lm.shape)
print(tpr_rf_lm.shape)

plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
plt.title('ROC curve-Training Set (With Imbalance), AUC: '+str(roc_auc))
plt.xlabel('False positive rate')
plt.plot([0, 1], [0, 1], 'k--')
# plt.legend('AUC:'str(round(roc_auc)))
plt.ylabel('True positive rate')
print(roc_auc)
plt.show()

confusion_matrx=confusion_matrix(Y_train,preds)
print(confusion_matrx)
sns.heatmap(confusion_matrx,cmap="YlGnBu",annot=True,linewidths=.5,fmt='d')
plt.show()

missclassifications=0
for i in range(0,2):
    for j in range(0,2):
        if i!=j:
            missclassifications+=confusion_matrx[i][j]

print(missclassifications)
print("missclassification rate:", missclassifications/(len(Y_train)))
print(rf_lm.predict_proba)
print("oob_error",(1-rf.oob_score_))
