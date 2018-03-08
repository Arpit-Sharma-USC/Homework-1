########################### 2d Simple RF test set with no imbalance ############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

import proc as prc
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc

df_train=pd.read_csv("F:/INF552/hw3/aps_failure_training_set.csv")#skiprows=20,index_col=21)
df_test=pd.read_csv("C:/Users/Shanu/PycharmProjects/APS/aps_failure_test_set.csv")

# Separate majority and minority classes
df_majority = df_train[df_train.classs == "neg"]
df_minority = df_train[df_train.classs == "pos"]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=59000,  # to match majority class 59000
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
print(df_upsampled.classs.value_counts())



# Separate majority and minority classes
df_majority_test = df_test[df_test.classs == "neg"]
df_minority_test = df_test[df_test.classs == "pos"]

# Upsample minority class
df_minority_upsampled_test = resample(df_minority_test,
                                 replace=True,  # sample with replacement
                                 n_samples=15625,  # to match majority class 59000
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled_test = pd.concat([df_majority_test, df_minority_upsampled_test])

# Display new class counts
print(df_upsampled_test.classs.value_counts())



df_upsampled.replace('na',0,inplace=True)
df_upsampled_test.replace('na',0,inplace=True)

# df_train.replace('neg',0,inplace=True)
# df_test.replace('pos',1,inplace=True)
X_train=df_upsampled.values[:,1:171]
Y_train=df_upsampled.values[:,:1]


X_test=df_upsampled_test.values[:,1:171]
Y_test=df_upsampled_test.values[:,:1]

model=RandomForestClassifier(max_depth=5)
model.fit(X_train,Y_train.ravel())

model_prob = model.predict_proba(X_test)
score=log_loss(Y_test,model_prob)
# score_mean=mean_squared_error(Y_test,model.predict(X_test))
print("Score:",score)
# print("MSE:",score_mean)
print("model_prob",model_prob)
model_prob=model_prob.reshape(1,-1)
#
rf = RandomForestClassifier(max_depth=3,oob_score=True)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, Y_train.ravel())
rf_enc.fit(rf.apply(X_train))
model_used=rf_lm.fit(rf_enc.transform(rf.apply(X_test)), Y_test.ravel())
preds=rf.predict(X_test)
print(type(model_used))

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(Y_test, y_pred_rf_lm,pos_label='pos')
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)

plt.plot(fpr_rf_lm, tpr_rf_lm, label=str(roc_auc))
plt.title('ROC curve-Test (With No Imbalance), AUC: '+str(roc_auc))
plt.xlabel('False positive rate')
plt.plot([0, 1], [0, 1], 'k--')
# plt.legend('AUC:'str(round(roc_auc)))
plt.ylabel('True positive rate')
print(roc_auc)
# RANDOM_STATE = 123
#
# ensemble_clfs = [
#     ("RandomForestClassifier, max_features='sqrt'",
#         RandomForestClassifier(warm_start=True, oob_score=True,
#                                max_features="sqrt",
#                                random_state=RANDOM_STATE)),
#     ("RandomForestClassifier, max_features='log2'",
#         RandomForestClassifier(warm_start=True, max_features='log2',
#                                oob_score=True,
#                                random_state=RANDOM_STATE)),
#     ("RandomForestClassifier, max_features=None",
#         RandomForestClassifier(warm_start=True, max_features=None,
#                                oob_score=True,
#                                random_state=RANDOM_STATE))
# ]
#
# # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
# error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
#
# # Range of `n_estimators` values to explore.
# min_estimators = 15
# max_estimators = 175
#
# for label, clf in ensemble_clfs:
#     for i in range(min_estimators, max_estimators + 1):
#         clf.set_params(n_estimators=i)
#         clf.fit(X_train, Y_train.ravel())
#
#         # Record the OOB error for each `n_estimators=i` setting.
#         oob_error = 1 - clf.oob_score_
#         error_rate[label].append((i, oob_error))
#
# # print("Score:",model_used.oob_score_)
# # error=1-model_used.oob_score_
# for label, clf_err in error_rate.items():
#     xs, ys = zip(*clf_err)
#     print("XS:",xs,"\t YS:",ys)
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

print("OOB Error:",(1-rf.oob_score_))
#