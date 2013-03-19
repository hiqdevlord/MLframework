'''
Created on Feb 18, 2013

@author: Ash Booth
'''
import sys
sys.path.insert(0, '/Users/user/git/scikit-learn')

import numpy as np
from sklearn import datasets
from sklearn import cross_validation as cv

from classification import Classification
from regression import Regression

if __name__ == '__main__':
    
#    # get classification data
#    digits = datasets.load_digits()
#    inputs = digits.data
#    targets = digits.target
#    
#    # get regression data
#    prices = datasets.load_boston()
#    inputs = prices.data
#    targets = prices.target

    full_data = np.genfromtxt("stock_metrics.csv",delimiter=',')
    inputs = full_data[:,:-1]
    targets  = full_data[:,-1]
    

    # Scale Features
    from sklearn import preprocessing
    scaler = preprocessing.Scaler().fit(inputs)
    input_digits = scaler.transform(inputs)
    
#    # Normalize Features
#    normalizer = preprocessing.Normalizer().fit(inputs)
#    inputs = normalizer.transform(inputs)
    
    # Split training and test sets
    train_features, test_features, train_targets, test_targets = cv.train_test_split(inputs, 
                                                                                     targets, test_size=0.3, random_state=1)

    print train_features.shape
    print train_targets.shape
    
    clf = Classification(True, train_features, train_targets, test_features, test_targets,
                                      PCA = False, Logit = True,
                                      SVC = True, NuSVC = True, SGDC = True,
                                      KNNC=  True, RNNC = True,
                                      GaussNB = False, MultiNB = False, BernNB = False,
                                      DTC = True, RFC = True, ETC = True, GBC = True, #adac = True, 
                                      LDA = True, QDA = True)
    
#    reg = Regression(True, train_features, train_targets, test_features, test_targets, PCA = False,
#                      linreg = True, ridge = False, lasso = False,
#                      svr = False, nusvr = False, 
#                      KNNR = False, RNNR = False,
#                      gausp = False, dtr = True, rfr = True, etr = True, gbr = True, adar = True)
#    
#    
    clf.fit_models(True)
    clf.test_models()
    
    clf.print_results()
