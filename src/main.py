'''
Created on Feb 18, 2013

@author: Ash Booth
'''

import numpy as np
from sklearn import datasets
from classification import Classification

if __name__ == '__main__':
    
    # get data
    digits = datasets.load_digits()
    
    input_digits = digits.data
    target_digits = digits.target
    input_train, input_test = np.array_split(input_digits[:-1],2) 
    target_train, target_test = np.array_split(target_digits[:-1],2)
    
    
    # Scale test
    from sklearn import preprocessing
    scaler = preprocessing.Scaler().fit(input_train)
    new_input_train = scaler.transform(input_train)
    new_input_test = scaler.transform(input_test)
    
    # Norm test
    normalizer = preprocessing.Normalizer().fit(input_train)
    new_input_train = normalizer.transform(input_train)
    new_input_test = normalizer.transform(input_test)
    
    class_pca = Classification(True, input_train, target_train, 
                                      input_test, target_test,
                                      PCA = True,
                                      Logit = True,
                                      SVC = True,
                                      NuSVC = True,
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
    
    class_no = Classification(True, new_input_train, target_train, 
                                      new_input_test, target_test,
                                      PCA = True,
                                      Logit = True,
                                      SVC = True,
                                      NuSVC = True,
                                      SGDC = True,
                                      KNNC=  True,
                                      RNNC = True,
                                      GaussNB = False,
                                      MultiNB = False, 
                                      BernNB = False,
                                      DTC = True,
                                      RFC = True,
                                      ETC = True,
                                      GBC = True,
                                      LDA = True,
                                      QDA = False)
    
    class_pca.fit_models(True)
    class_pca.test_models()
    
    class_no.fit_models(True)
    class_no.test_models()
    
    class_pca.print_results()
    class_no.print_results()
    