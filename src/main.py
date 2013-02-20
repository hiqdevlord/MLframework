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
    
    
    class_test = Classification(input_digits, target_digits,
                 input_test, target_test,pca=True,logit=True)
    
    class_test.run()