'''
Created on Feb 27, 2013

@author: Ash Booth
'''

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

from sklearn import decomposition

from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNet
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble.forest import RandomForestRegressor
import sklearn

class Regression(object):
    '''
    classdocs
    '''


    def __init__(self, verbose,
                 inputs_train_val, targets_train_val,
                 inputs_test, targets_test, 
                 PCA = False,
                 linreg = False, ridge = False, lasso = False,
                 svr = False, nusvr = False, 
                 KNNR = False, RNNR = False,
                 gausp = False,
                 dtr = False, rfr = False, etr = False, gbr = False, adar = False):
        '''
        Constructor
        '''
        self.NFOLDS = 10
        self.inputs_train_val = inputs_train_val
        self.targets_train_val = targets_train_val
        self.inputs_test = inputs_test
        self.targets_test = targets_test
        self.PCA = PCA
        
        self.estimators = []
        
        if linreg: self.estimators.append(self.__gen_linreg_estimator(verbose))
        if ridge: self.estimators.append(self.__gen_ridge_estimator(verbose))
        if lasso: self.estimators.append(self.__gen_lassocv_estimator(verbose))
        if svr: self.estimators.append(self.__gen_svr_estimator(verbose))
        if nusvr: self.estimators.append(self.__gen_nusvr_estimator(verbose))
        if KNNR: self.estimators.append(self.__gen_knnr_estimator(verbose))
        if RNNR: self.estimators.append(self.__gen_rnnr_estimator(verbose))
        if gausp: self.estimators.append(self.__gen_gausp_estimator(verbose))
        if dtr: self.estimators.append(self.__gen_dtr_estimator(verbose))
        if rfr: self.estimators.append(self.__gen_rfr_estimator(verbose))
        if etr: self.estimators.append(self.__gen_etr_estimator(verbose))
        if gbr: self.estimators.append(self.__gen_gbr_estimator(verbose))
        if adar: self.estimators.append(self.__gen_adar_estimator(verbose))
        self.outdata = []
        

    def __gen_linreg_estimator(self,verbose):
        linreg = LinearRegression()
        if self.PCA:
            description = "logistic regression with PCA"
            if verbose: print "generating linear regressor with PCA..."
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('linreg', linreg)])
            n_components = [20, 40, 64]
            normalize = [True, False]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     linreg__normalize=normalize), n_jobs=-1, 
                                     verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else: 
            description = "logistic regression"
            if verbose: print "generating logistic regression grid search..."
            pipe = Pipeline(steps=[('linreg', linreg)])
            normalize = [True, False]
            estimator = GridSearchCV(pipe,
                                     dict(linreg__normalize=normalize), n_jobs=-1, 
                                     verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    def __gen_ridge_estimator(self,verbose):
        ridge = Ridge(copy_X=True)
        if self.PCA:
            description = "ridge regression with PCA"
            if verbose: print "generating {} grid search...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('ridge', ridge)])
            n_components = [20, 40, 64]
            normalize = [True, False]
            alpha = [0.2, 0.5, 1.0, 1.5]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     ridge__normalize=normalize, ridge__alpha=alpha), n_jobs=-1, 
                                     verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else: 
            description = "ridge regression"
            if verbose: print "generating {} grid search...".format(description)
            pipe = Pipeline(steps=[('ridge', ridge)])
            normalize = [True, False]
            alpha = [0.2, 0.5, 1.0, 1.5]
            estimator = GridSearchCV(pipe,
                                     dict(ridge__normalize=normalize, ridge__alpha=alpha), n_jobs=-1, 
                                     verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    def __gen_lassocv_estimator(self,verbose):
        lasso = LassoCV()
        if self.PCA:
            description = "lasso regression with PCA"
            if verbose: print "generating {} grid search...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('lasso', lasso)])
            n_components = [20, 40, 64]
            cv = [cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS)]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                     lasso__cv=cv))
        else: 
            description = "lasso regression"
            if verbose: print "generating {} grid search...".format(description)
            pipe = Pipeline(steps=[('lasso', lasso)])
            cv = [cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS)]
            estimator = GridSearchCV(pipe,
                                     dict(lasso__cv=cv))
        return [estimator, description]


    
    def __gen_svr_estimator(self,verbose):
        svr = SVR()
        if self.PCA:
            description = "SVR with PCA"
            if verbose: print "generating {} grid search...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('svr', svr)])
            n_components = [20, 40, 64]
            epsilon = [np.linspace(0.05,0.2,4)]
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe, dict(pca__n_components=n_components,svr__C=Cs,svr__kernel=kernels,svr__epsilon=epsilon), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else:
            description = "SVR"
            if verbose: print "generating {} grid search...".format(description)
            pipe = Pipeline(steps=[('svr', svr)])
            epsilon = [np.linspace(0.05,0.2,4)]            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            Cs = np.logspace(-4, 4, 3)
            estimator = GridSearchCV(pipe, dict(svr__C=Cs, svr__kernel=kernels,svr__epsilon=epsilon), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    def __gen_nusvr_estimator(self,verbose):
        nusvr = NuSVR()
        if self.PCA:
            description= "NuSVR with PCA"
            if verbose: print "generating {} grid search...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('nusvr', nusvr)])
            n_components = [20, 40, 64]
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            nu = [0.2, 0.4, 0.6, 0.8]
            estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, nusvr__nu=nu,nusvr__kernel=kernels), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else:
            description= "NuSVR"
            if verbose: print "generating {} grid search...".format(description)
            pipe = Pipeline(steps=[('nusvr', nusvr)])            
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            nu = [0.2, 0.4, 0.6, 0.8]
            estimator = GridSearchCV(pipe, dict(nusvr__nu=nu, nusvr__kernel=kernels), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    
    def __gen_knnr_estimator(self,verbose):
        knn = KNeighborsRegressor()
        if self.PCA:
            description = "KNNRegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])
            n_components = [20, 40, 64]
            n_neighbors = [3,5,9]
            estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, knn__n_neighbors=n_neighbors), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else:
            description = "KNNRegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('knn', knn)])
            n_neighbors = [3,5,9]
            estimator = GridSearchCV(pipe, dict(knn__n_neighbors=n_neighbors), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    def __gen_rnnr_estimator(self,verbose):
        rnn = RadiusNeighborsRegressor()
        if self.PCA:
            description = "RNNRegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('rnn', rnn)])
            n_components = [20, 40, 64]
            algorithm = ['auto'] 
            estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, rnn__algorithm=algorithm), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else:
            description = "RNNRegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('rnn', rnn)])
            algorithm = ['auto']
            estimator = GridSearchCV(pipe, dict(rnn__algorithm=algorithm), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    def __gen_gausp_estimator(self,verbose):
        gausp = GaussianProcess()
        if self.PCA:
            description = "Gaussian Process with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('gausp', gausp)])
            n_components = [20, 40, 64]
            regr = ['constant', 'linear']
            estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, gausp__regr=regr), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        else:
            description = "Gaussian Process"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('gausp', gausp)])
            regr = ['constant', 'linear']
            estimator = GridSearchCV(pipe, dict(gausp__regr=regr), 
                                     n_jobs=-1, verbose=1, cv=cross_validation.KFold(self.inputs_train_val.shape[0],k=self.NFOLDS))
        return [estimator, description]
    
    

    def __gen_dtr_estimator(self,verbose):
        dtr = DecisionTreeRegressor()
        if self.PCA:
            description = "DTRegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('dtr', dtr)])
            n_components = [20, 40, 64]
            max_depth = [None, 3, 5, 7]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                          dtr__max_depth=max_depth))
        else:
            description = "DTRegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('dtr', dtr)])
            max_depth = [None, 3, 5, 7]
            estimator = GridSearchCV(pipe,
                                     dict(dtr__max_depth=max_depth))
        return [estimator, description]
    

    def __gen_rfr_estimator(self,verbose):
        rfr = RandomForestRegressor(n_jobs=1)
        if self.PCA:
            description = "RFRegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('rfr', rfr)])
            n_components = [20, 40, 64]
            n_estimators = [5, 10, 15]
            max_depth = [None, 3, 5, 7]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                          rfr__max_depth=max_depth,
                                          rfr__n_estimators=n_estimators))
        else:
            description = "RFRegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('rfr', rfr)])
            n_estimators = [5, 10, 15]
            max_depth = [None, 3, 5, 7]
            estimator = GridSearchCV(pipe,
                                     dict(rfr__max_depth=max_depth,
                                          rfr__n_estimators=n_estimators))
        return [estimator, description]


    def __gen_etr_estimator(self,verbose):
        etr = ExtraTreesRegressor(n_jobs=1)
        if self.PCA:
            description = "ETRegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('etr', etr)])
            n_components = [20, 40, 64]
            n_estimators = [5, 10, 15]
            max_depth = [None, 3, 5, 7]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                          etr__max_depth=max_depth,
                                          etr__n_estimators=n_estimators))
        else:
            description = "ETRegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('etr', etr)])
            n_estimators = [5, 10, 15]
            max_depth = [None, 3, 5, 7]
            estimator = GridSearchCV(pipe,
                                     dict(etr__max_depth=max_depth,
                                          etr__n_estimators=n_estimators))
        return [estimator, description]
    
    
    def __gen_gbr_estimator(self,verbose):
        gbr = GradientBoostingRegressor(n_estimators=100)
        if self.PCA:
            description = "GBRegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('gbr', gbr)])
            n_components = [20, 40, 64]
            max_depth = [1, 3, 5]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                          gbr__max_depth=max_depth))
        else:
            description = "GBRegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('gbr', gbr)])
            max_depth = [1, 3, 5]
            estimator = GridSearchCV(pipe,
                                     dict(gbr__max_depth=max_depth))
        return [estimator, description]

    def __gen_adar_estimator(self,verbose):
        adar = AdaBoostRegressor()
        if self.PCA:
            description = "ADARegressor with PCA"
            if verbose: print "generating {}...".format(description)
            pca = decomposition.PCA()
            pipe = Pipeline(steps=[('pca', pca), ('adar', adar)])
            n_components = [20, 40, 64]
            n_estimators = [15, 50, 100, 200]
            learning_rate = [0.8,1.0,1.2]
            estimator = GridSearchCV(pipe,
                                     dict(pca__n_components=n_components,
                                          adar__learning_rate = learning_rate, adar__n_estimators=n_estimators))
        else:
            description = "ADARegressor"
            if verbose: print "generating {}...".format(description)
            pipe = Pipeline(steps=[('adar', adar)])
            n_estimators = [15, 50, 100, 200]
            learning_rate = [0.8,1.0,1.2]
            estimator = GridSearchCV(pipe,
                                     dict(adar__n_estimators=n_estimators,adar__learning_rate = learning_rate))
        return [estimator, description]


    def fit_models(self, verbose):
        print "\nfitting models..."
        for e in self.estimators:
            if verbose: print "fitting {}...".format(e[1])
            e[0].fit(self.inputs_train_val,self.targets_train_val)
        
        
    def test_models(self):
        print "\ntesting models..."
        for e in self.estimators:
            cv_score = e[0].score(self.inputs_train_val, self.targets_train_val)
            test_score = e[0].score(self.inputs_test, self.targets_test)
            self.outdata.append([e[1],cv_score,test_score])
    
    def write_data(self,verbose):
        print "\nwriting data..."
        import csv
        if self.PCA: filenames = ["scores_pca.csv", "params_pca.txt"]
        else: filenames = ["scores.csv", "params.txt"]
        with open(filenames[0], 'wb') as csvfile:
            scores = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            scores.writerows(self.outdata)
        param_file = open(filenames[1], 'wb')
        for e in self.estimators:
            param_file.write(e[1])
            param_file.write("{}".format(e[0]))
            param_file.write("\n\n")
        param_file.close()
        
    def print_results(self):
        temp_data = self.outdata
        temp_data.insert(0, ["Algorithm", "Validation Score", "Test Score\n"])
        s = [[str(e) for e in row] for row in self.outdata]
        lens = [len(max(col, key=len)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print '\n'
        print '\n'.join(table)   
            

        