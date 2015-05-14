#!/usr/bin/python
import xgboost as xgb

import numpy as np
import pandas as pd
import time,datetime
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from util import *


#np.random.seed(1337) # for reproducibility
'''
#parameters
golden_ratio=1.61803398874989484820458683436563811772030917980576286213544862270526046281890
lr =  .097070003398874989484820450683436563811772030917980576286213544862270526046281890
max_depth = 15
silent =1
objective =  'multi:softprob'
num_class = 9 
subsample = .5
gamma = golden_ratio
min_child_weight = 1
max_delta_step =  1
colsample_bytree = 1
base_score = .5
nthread = 4
num_round =1000
'''





def otto_xgboost(lr,max_depth,silent,objective,num_class,subsample,gamma,min_child_weight,max_delta_step,colsample_bytree,base_score,nthread,num_round):


#record time
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')



    X,y,X_test=read_data()
    y=y-1
    dataset=make_dataset(X,y)
    Xtrain,ytrain,Xcross,ycross=make_cross_validation(dataset,0.90)

    dtrain = xgb.DMatrix( Xtrain, label=ytrain)
    dcross = xgb.DMatrix( Xcross, label=ycross)
    dtest = xgb.DMatrix(X_test)

    param = {'bst:max_depth':max_depth,'bst:eta':lr, 'silent':silent, 'objective':objective,'num_class':num_class,'subsample':subsample,'gamma':golden_ratio,
     'min_child_weight':min_child_weight, 'max_delta_step':max_delta_step,  'colsample_bytree':colsample_bytree,'base_score':base_score, 'seed':1337}
    param['nthread'] = nthread

    plst = param.items()
    plst += [('eval_metric', 'mlogloss')]
    evallist  = [(dcross,'eval'), (dtrain,'train')]

    evals_result={}
    bst = xgb.train( plst, dtrain, num_round, evallist ,evals_result=evals_result)
    y_pred = bst.predict(dtest)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    #same parameters in dictionary
    print float(evals_result['eval'][num_round-1])-float(evals_result['train'][num_round-1])
    threshold = 0.430
    if float(evals_result['eval'][num_round-1])<threshold:
        send_me_email(st,threshold)
        write_to_file(y_pred,param,evals_result,st)
        bst.save_model('predict'+str(st)+'model')
    


if __name__ == '__main__':
    golden_ratio=1.618033988749
                  
    gamma = golden_ratio
    for lr in np.arange(.098090003398,.099090003398,.00001):
        print ("begin"+str(lr))
        otto_xgboost(lr = lr,
                        max_depth = 15,
                        silent =1,
                        objective =  'multi:softprob',
                        num_class = 9 ,
                        subsample = .5,
                        gamma = gamma,
                        min_child_weight = 1,
                        max_delta_step =  1,
                        colsample_bytree = .9,
                        base_score = .5,
                        nthread = 4,
                        num_round =500)