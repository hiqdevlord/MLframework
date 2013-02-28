'''
Created on Feb 28, 2013

@author: Ash Booth
'''

import numpy as np
from sklearn import datasets
from classification import Classification
from traits.tests import test_target

if __name__ == '__main__':
    full_data = np.genfromtxt("train_data.csv",delimiter=',')
    
    # Scale Features
    from sklearn import preprocessing
    scaler = preprocessing.Scaler().fit(full_data[:,:-1])
#    full_data[:,:-1] = scaler.transform(full_data[:,:-1])

    # Shuffle data
    np.random.shuffle(full_data)
    
    # Split training and test sets
    bp = len(full_data)*4/5
    train_features = full_data[:bp,:-1]
    train_targets = full_data[:bp,-1]
    test_features = full_data[bp:,:-1]
    test_targets = full_data[bp:,-1]
    
    class_nopca = Classification(True, train_features, train_targets, 
                                      test_features, test_targets,
                                      PCA = False,
                                      Logit = True,
                                      SVC = True,
                                      NuSVC = False,
                                      SGDC = True,
                                      KNNC=  True,
                                      RNNC = True,
                                      GaussNB = True,
                                      MultiNB = False, 
                                      BernNB = False,
                                      DTC = True,
                                      RFC = True,
                                      ETC = True,
                                      GBC = True,
                                      LDA = True,
                                      QDA = False)
    
    class_nopca.fit_models(True)
    class_nopca.test_models()
    class_nopca.write_data(False)
    class_nopca.print_results()