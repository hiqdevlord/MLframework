'''
Created on Feb 19, 2013

@author: Ash Booth
'''

import os
import gzip
import cPickle as pickle
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import datasets, svm

print '... loading data'
digits = datasets.load_digits()
inputs = digits.data
targets = digits.target

train_inputs, val_inputs, test_inputs = np.array_split(inputs,3)
train_targets, val_targets, test_targets = np.array_split(targets,3)


svc = svm.SVC(C=1, kernel='linear')
tr_v_in = np.concatenate((train_inputs,val_inputs),axis=0)
tr_v_targ = np.concatenate((train_targets,val_targets),axis=0)

kfold = cross_validation.KFold(n=len(tr_v_in), k=3)
scores = cross_validation.cross_val_score(svc, tr_v_in, tr_v_targ, cv=kfold, n_jobs=-1)

# sklearn provides an object that, given data, computes the score during the fit of an estimator 
# on a parameter grid and chooses the parameters to maximize the cross-validation score. This object 
# takes an estimator during the construction and exposes an estimator API:
from sklearn.grid_search import GridSearchCV
gammas = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), n_jobs=-1)
clf.fit(tr_v_in, tr_v_targ)
print clf.best_score_
# Prediction performance on test set is not as good as on train set
print clf.score(test_inputs, test_targets)

# Fantastic use of grid search
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)


## Train a model
#start_time = time.clock()
#clf = svm.SVC(gamma=0.001, C=100.)
#clf.fit(train_set_inputs, train_set_targets) 
#end_time = time.clock()
#
## Save a trained model with Pickle
#output = open('data.pkl', 'wb')
#s = pickle.dumps(clf,output)
#clf2 = pickle.loads(s)
#clf2.predict(train_set_inputs[0])
#
#
#print 'The code ran for %d seconds' % (end_time - start_time)