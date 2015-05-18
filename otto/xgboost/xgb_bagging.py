 #!/usr/bin/env python
import sys
import time
import logging
import numpy as np
import csv
import random
import xgboost

from sklearn import preprocessing
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from util import *


np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
logging.basicConfig(format="%(message)s", level=logging.DEBUG, stream=sys.stdout)
train_data_array,train_label_array,X_test=read_data()
dataset=make_dataset(train_data_array,train_label_array)
X_train,y_train,X_cross,y_cross=make_cross_validation(dataset,0.9)

 
clf = xgboost.XGBClassifier(max_depth=15, learning_rate=.080090003398, n_estimators=600, silent=True, objective='multi:softprob',
                 nthread=4, gamma=1.2145435, min_child_weight=1, max_delta_step=1, subsample=.5, colsample_bytree=.9,
                 base_score=0.5, seed=1331)
'''
#adaptor to use gradient boosting
#bagging don't need this adaport in scikit-learn
class Adaptor(object):
    def __init__(self, est):    
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)
    def fit(self, X, y):
        self.est.fit(X, y)

clf = Adaptor(clf)

'''
from sklearn import ensemble
gbc =  ensemble.BaggingClassifier(base_estimator=clf, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True,
 bootstrap_features=False, oob_score=False, n_jobs=-1, random_state=None, verbose=1)
print("training ...")
gbc.fit(X_train,y_train)
print("done")
y_train_pred=gbc.predict(X_train)
result=(y_train_pred==y_train)*1
print("Training accuracy "+str(sum(result)/float(y_train.shape[0])))
y_pred_labels=gbc.predict(X_cross)
result=(y_pred_labels==y_cross)*1
print("Cross accuracy "+str(sum(result)/float(y_cross.shape[0])))
pred = gbc.predict_proba(X_test)
write_to_file(pred)

