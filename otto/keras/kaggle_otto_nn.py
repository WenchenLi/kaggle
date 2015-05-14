from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.regularizers import l2,l1
from keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Adam
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.

    Compatible Python 2.7-3.4 

    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.

    Best validation score at epoch 21: 0.4881 

    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
'''

np.random.seed(1337) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        
        #add log to data
        X = np.log(1+X)
        
        scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    #add gaussian noise    
    mu, sigma = 0, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma)
    #X = X + s
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))


print("Loading data...")
X, labels = load_data('train.csv', train=True)
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

X_test, ids = load_data('test.csv', train=False)
X_test, _ = preprocess_data(X_test, scaler)















#X=np.hstack((np.hstack((np.real(np.fft.fft(X,axis=-1)),np.imag(np.fft.fft(X,axis=-1)))),X))
#X_test = np.hstack((np.hstack((np.real(np.fft.fft(X_test,axis=-1)),np.imag(np.fft.fft(X_test,axis=-1)))),X_test))


clf = ExtraTreesClassifier()
X= clf.fit(X, y).transform(X)
X_test=clf.transform(X_test)

#drop features
#features= [34,48,16,39,62,68,60,67,22,18,14,11,43,87,75,42,59,45,15,55,26,1,56,38,64,70,29,85,32,50,21,40,69,9,86,72,91,36,33,90,41,73,23,74,93,53,77]
#feature =[ item-1 for item in features]
'''
import random 
feature = range(93)
feature = random.sample(feature,60)
X=X[:,feature]
X_test=X_test[:,feature]
'''



nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")


hidden_units_number=128
l2_lambda = .001
dropout = 0.5
initialization = 'lecun_uniform'

model = Sequential()
model.add(Dense(dims, hidden_units_number, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(.2))

model.add(Dense(hidden_units_number, hidden_units_number, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(dropout))

model.add(Dense(hidden_units_number, hidden_units_number, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(dropout))

model.add(Dense(hidden_units_number, hidden_units_number, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(dropout))

model.add(Dense(hidden_units_number, hidden_units_number, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(dropout))

model.add(Dense(hidden_units_number, hidden_units_number, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(dropout))

model.add(Dense(hidden_units_number, hidden_units_number, init=initialization,W_regularizer = l1(l2_lambda)))
model.add(PReLU((hidden_units_number,)))
model.add(BatchNormalization((hidden_units_number,)))
model.add(Dropout(dropout))

model.add(Dense(hidden_units_number, nb_classes, init=initialization,W_regularizer = l2(l2_lambda)))
model.add(Activation('softmax'))

#optimizers
''''
sgd=Adagrad(lr=0.01, epsilon=1e-6)

sgd=Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)

sgd=RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)

sgd=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8)

sgd = SGD(lr=1, momentum=0.9, decay=1, nesterov=True)

'''

sgd = SGD(lr=1.61803398874989484820458683436563811772030917980576286213544862270526046281890, momentum=0.95, decay=1, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)




print("Training model...")

model.fit(X, y, nb_epoch=10000, batch_size=1024, validation_split=0.1,shuffle = True)

print("Generating submission...")

proba = model.predict_proba(X_test)
make_submission(proba, ids, encoder, fname='keras-otto.csv')

