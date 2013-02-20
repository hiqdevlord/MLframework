'''
Created on Feb 20, 2013

@author: Ash Booth

TODO:
- allow ability to check parameterisation of the best models (results)
'''

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn import decomposition
from sklearn.linear_model import LogisticRegression #, Perceptron
from sklearn.svm import SVC, NuSVC, LinearSVC

class Classification(object):
    '''
    classdocs
    '''


    def __init__(self,
                 inputs_train_val, targets_train_val,
                 inputs_test, targets_test, 
                 pca = False,
                 logit = False, perceptron = False, 
                 SVC = False, NuSVC = False, LinearSVC = False ):
        '''
        Constructor
        '''
        self.inputs_train_val = inputs_train_val
        self.targets_train_val = targets_train_val
        self.inputs_test = inputs_test
        self.targets_test = targets_test
        self.pca = pca
        self.estimators = []
        if logit: self.estimators.append(self.__gen_logit_estimator())
        

    def __gen_logit_estimator(self):
        logistic = LogisticRegression()
        if self.pca:
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
            n_components = [20, 40, 64]
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     logistic__C=Cs))
        else: 
            pipe = Pipeline(steps=[('logistic', logistic)])
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(logistic__C=Cs))
        return [estimator, "logistic regression"]
    
    def __gen_svc_estimator(self):
        svc = SVC()
        if self.pca:
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
            n_components = [20, 40, 64]
            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     svc__C=Cs,svc__kernel=kernels))
        else:
            pipe = Pipeline(steps=[('svc', svc)])
            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(svc__C=Cs, svc__kernel=kernels))
        return [estimator, "support vector classifier"]

    def run(self):
        self.estimators[0][0].fit(self.inputs_train_val,self.targets_train_val)
        print "Num components chosen = {}\nVal for C chosen = {}".format(self.estimators[0][0].best_estimator_.named_steps['pca'].n_components,
                                                                         self.estimators[0][0].best_estimator_.named_steps['logistic'].C)
#        for e in self.estimators:
#            e.fit()
        # for all estimators check on test set
        # output results
        
    