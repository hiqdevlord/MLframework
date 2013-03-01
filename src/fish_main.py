'''
Created on Feb 28, 2013

@author: Ash Booth
'''

import numpy as np
from sklearn import datasets
from classification import Classification
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from time import time
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

if __name__ == '__main__':
    full_data = np.genfromtxt("train_data.csv",delimiter=',')
    
    # Scale Features
    from sklearn import preprocessing
    scaler = preprocessing.Scaler().fit(full_data[:,:-1])
#    full_data[:,:-1] = scaler.transform(full_data[:,:-1])

    # Split training and test sets
    train_features, test_features, train_targets, test_targets = cross_validation.train_test_split(full_data[:,:-1], 
                                                                                                   full_data[:,-1], test_size=0.2, random_state=1)
    
    
    
    def a_bomb(tr_features, tr_targets, te_features, te_targets, verbose):
        class_nopca = Classification(True, tr_features, tr_targets, 
                                      te_features, te_targets,
                                      PCA = False,
                                      Logit = True,
                                      SVC = True,
                                      NuSVC = False,
                                      SGDC = True,
                                      KNNC=  True,
                                      RNNC = False,
                                      GaussNB = False,
                                      MultiNB = True, 
                                      BernNB = False,
                                      DTC = True,
                                      RFC = True,
                                      ETC = True,
                                      GBC = True,
                                      LDA = True,
                                      QDA = False)
    
        class_nopca.fit_models(verbose)
        class_nopca.test_models()
        class_nopca.write_data(verbose)
        class_nopca.print_results()
    
    def workflow(pipe,params,inputs,targets):
        # find the best parameters for both the classifying and the
        # classifier
        cv = cross_validation.LeaveOneOut(inputs.shape[0])
        grid_search = GridSearchCV(pipe, params, n_jobs=-1, verbose=1, cv=cv)
    
        print "Performing grid search..."
        print "pipeline:", [name for name, _ in pipe.steps]
        print "parameters:"
        pprint(parameters)
        t0 = time()
        grid_search.fit(inputs, targets)
        print "done in %0.3fs" % (time() - t0)
        print
    
        print "Best CV score: %0.3f" % grid_search.best_score_
        print "Best parameters set:"
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print "\t%s: %r" % (param_name, best_parameters[param_name])
    
    
    ###############################################################################
    
    # Drop the A-bomb
    #a_bomb(train_features, train_targets, test_features, test_targets, True)
    
    ###############################################################################
    
    # Exploring Logistic regression
    
    pipeline = Pipeline([
              ('clf', LogisticRegression()),
              ])
    
    parameters = {
            'clf__C': (np.logspace(-4, 4, 7)),
            'clf__penalty': ('l1', 'l2'),
    }
    
    #workflow(pipeline, parameters, train_features, train_targets)
    
    clf = LogisticRegression(penalty='l1', C=1.0)
    clf.fit(train_features, train_targets)
    print clf.score(test_features, test_targets)
    
    preds = clf.predict(test_features)
    
    print(classification_report(test_targets, preds, labels = [1,2,3,4,5,6],target_names=['SB', 'SS', 'ST', 'T', 'TB', 'U']))
    
    labels = ['SB', 'SS', 'ST', 'T', 'TB', 'U']
    cnf = confusion_matrix(test_targets, preds, labels=[1,2,3,4,5,6])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cnf)
    fig.colorbar(cax)
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    plt.show()
    
    