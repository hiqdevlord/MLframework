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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

class Classification(object):
    '''
    classdocs
    '''


    def __init__(self, verbose,
                 inputs_train_val, targets_train_val,
                 inputs_test, targets_test, 
                 PCA = False,
                 Logit = False, perceptron = False, SGDC = True,
                 SVC = False, NuSVC = False, 
                 KNNC = False, RNNC = False,
                 GaussNB = False, MultiNB = False, BernNB = False,
                 DTClassifier = False):
        '''
        Constructor
        '''
        self.inputs_train_val = inputs_train_val
        self.targets_train_val = targets_train_val
        self.inputs_test = inputs_test
        self.targets_test = targets_test
        self.PCA = PCA
        
        self.estimators = []
        
        if Logit: self.estimators.append(self.__gen_logit_estimator(verbose))
        if SVC: self.estimators.append(self.__gen_svc_estimator(verbose))
        if NuSVC: self.estimators.append(self.__gen_nusvc_estimator(verbose))
        if SGDC: self.estimators.append(self.__gen_sgd_estimator(verbose))
        if KNNC: self.estimators.append(self.__gen_knn_estimator(verbose))
        if RNNC: self.estimators.append(self.__gen_rnn_estimator(verbose))
        if GaussNB: self.estimators.append(self.__gen_gnb_estimator(verbose))
        if MultiNB: self.estimators.append(self.__gen_mnb_estimator(verbose))
        if BernNB: self.estimators.append(self.__gen_bnb_estimator(verbose))
        if DTClassifier: self.estimators.append(self.__gen_DTC_estimator(verbose))
        
        self.outdata = []
        

    def __gen_logit_estimator(self,verbose):
        logistic = LogisticRegression()
        if self.PCA:
            description = "logit classifier with PCA"
            if verbose: print "generating logit classifier grid search with PCA..."
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
            n_components = [20, 40, 64]
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     logistic__C=Cs))
        else: 
            description = "logit classifier"
            if verbose: print "generating logit classifier grid search..."
            pipe = Pipeline(steps=[('logistic', logistic)])
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(logistic__C=Cs))
        return [estimator, description]
    
    def __gen_svc_estimator(self,verbose):
        svc = SVC()
        if self.PCA:
            description = "SVC with PCA"
            if verbose: print "generating support vector classifier grid search with PCA..."
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
            n_components = [20, 40, 64]
            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     svc__C=Cs,svc__kernel=kernels))
        else:
            description = "SVC"
            if verbose: print "generating support vector classifier grid search..."
            pipe = Pipeline(steps=[('svc', svc)])            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe,
                                     dict(svc__C=Cs, svc__kernel=kernels))
        return [estimator, description]
    
    def __gen_nusvc_estimator(self,verbose):
        nusvc = NuSVC()
        if self.PCA:
            description= "NuSVC with PCA"
            if verbose: print "generating nu support vector classifier grid search with PCA..."
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('nusvc', nusvc)])
            n_components = [20, 40, 64]
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            nu = [0.2, 0.4, 0.6, 0.8]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     nusvc__nu=nu,nusvc__kernel=kernels))
        else:
            description= "NuSVC"
            if verbose: print "generating nu support vector classifier grid search..."
            pipe = Pipeline(steps=[('svc', nusvc)])            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            nu = np.linspace(0, 1, 5)
            estimator = GridSearchCV(pipe,
                                     dict(nusvc__nu=nu, nusvc__kernel=kernels))
        return [estimator, description]
    
    def __gen_sgd_estimator(self,verbose):
        sgd = SGDClassifier()
        if self.PCA:
            description = "SGDClassifier with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('sgd', sgd)])
            n_components = [20, 40, 64]
            loss = ['hinge', 'log', 'modified_huber']
            penalty = ['l2', 'l1', 'elasticnet']
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     sgd__loss=loss, sgd__penalty=penalty))
        else:
            description = "SGDClassifier"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('sgd', sgd)])
            loss = ['hinge', 'log', 'modified_huber']
            penalty = ['l2', 'l1', 'elasticnet']
            estimator = GridSearchCV(pipe,
                                     dict(sgd__loss=loss, sgd__penalty=penalty))
        return [estimator, description]
    
    def __gen_knn_estimator(self,verbose):
        knn = KNeighborsClassifier()
        if self.PCA:
            description = "KNNClassifier with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])
            n_components = [20, 40, 64]
            n_neighbors = [3,5,9]
            algorithm = ['ball_tree', 'kd_tree', 'brute']
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     knn__n_neighbors=n_neighbors, knn__algorithm=algorithm))
        else:
            description = "KNNClassifier"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('knn', knn)])
            n_neighbors = [3,5,9]
            algorithm = ['ball_tree', 'kd_tree', 'brute']
            estimator = GridSearchCV(pipe,
                                     dict(knn__n_neighbors=n_neighbors, knn__algorithm=algorithm))
        return [estimator, description]
    
    def __gen_rnn_estimator(self,verbose):
        rnn = RadiusNeighborsClassifier(outlier_label=-1)
        if self.PCA:
            description = "RNNClassifier with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('rnn', rnn)])
            n_components = [20, 40, 64]
            radius = [1.0]
            algorithm = ['auto'] # ['ball_tree', 'kd_tree', 'brute']
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     rnn__radius=radius, rnn__algorithm=algorithm))
        else:
            description = "RNNClassifier"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('rnn', rnn)])
            radius = [1.6]
            algorithm = ['auto'] #['ball_tree', 'kd_tree', 'brute']
            estimator = GridSearchCV(pipe,
                                     dict(rnn__radius=radius, rnn__algorithm=algorithm))
        return [estimator, description]
    
    def __gen_gnb_estimator(self,verbose):
        gnb = GaussianNB()
        if self.PCA:
            description = "GaussianNB with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('gnb', gnb)])
            n_components = [20, 40, 64]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components))
        else:
            description = "GaussianNB"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('gnb', gnb)])
            
            estimator = GridSearchCV(pipe)
        return [estimator, description]
    
    def __gen_mnb_estimator(self,verbose):
        mnb = MultinomialNB(fit_prior=True)
        if self.PCA:
            description = "MultinomialNB with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('mnb', mnb)])
            n_components = [20, 40, 64]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components))
        else:
            description = "MultinomialNB"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('mnb', mnb)])
            
            estimator = GridSearchCV(pipe)
        return [estimator, description]
    
    def __gen_bnb_estimator(self,verbose):
        bnb = BernoulliNB(fit_prior=True)
        if self.PCA:
            description = "BernoulliNB with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('bnb', bnb)])
            n_components = [20, 40, 64]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components))
        else:
            description = "BernoulliNB"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('bnb', bnb)])
            
            estimator = GridSearchCV(pipe)
        return [estimator, description]
    
    def __gen_dtc_estimator(self,verbose):
        dtc = DecisionTreeClassifier()
        if self.PCA:
            description = "DTClassifier with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('dtc', dtc)])
            n_components = [20, 40, 64]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components))
        else:
            description = "DTClassifier"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('dtc', dtc)])
            
            estimator = GridSearchCV(pipe)
        return [estimator, description]
    

    def fit_models(self, verbose):
        for e in self.estimators:
            if verbose: print "fitting {}...".format(e[1])
            e[0].fit(self.inputs_train_val,self.targets_train_val)
        
        
    def test_models(self):
        for e in self.estimators:
            cv_score = e[0].score(self.inputs_train_val, self.targets_train_val)
            test_score = e[0].score(self.inputs_test, self.targets_test)
            self.outdata.append([e[1],cv_score,test_score])
    
    def write_data(self):
        print "\n"
        for d in self.outdata: print d
#        print (self.estimators[0][0].best_estimator_)   
#        print "\nPCA with logistic regression:"
#        print "Num components chosen = {}\nVal for C chosen = {}".format(self.estimators[0][0].best_estimator_.named_steps['pca'].n_components,
#                                                                         self.estimators[0][0].best_estimator_.named_steps['logistic'].C)
#        print "\nPCA with Support Vector Classifier:"
#        print "Num components chosen = {}\nVal for C chosen = {}\nkernel = {}".format(self.estimators[1][0].best_estimator_.named_steps['pca'].n_components,
#                                                                 self.estimators[1][0].best_estimator_.named_steps['svc'].C,
#                                                                 self.estimators[1][0].best_estimator_.named_steps['svc'].kernel)
#        print "\nPCA with nu Support Vector Classifier:"
#        print "Num components chosen = {}\nVal for nu chosen = {}\nkernel = {}".format(self.estimators[2][0].best_estimator_.named_steps['pca'].n_components,
#                                                                 self.estimators[2][0].best_estimator_.named_steps['nusvc'].nu,
#                                                                 self.estimators[2][0].best_estimator_.named_steps['nusvc'].kernel)
#        pass
#        
#        for e in self.estimators:
#            e.fit()
        # for all estimators check on test set
        # output results
        
    