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
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
from sklearn.svm import SVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

if __name__ == '__main__':
    full_data = np.genfromtxt("train_data.csv",delimiter=',')
    
    # Scale Features
    from sklearn import preprocessing
    scaler = preprocessing.Scaler().fit(full_data[:,:-1])
    full_data[:,:-1] = scaler.transform(full_data[:,:-1])

    # Split training and test sets
    train_features, test_features, train_targets, test_targets = cross_validation.train_test_split(full_data[:,:-1], 
                                                                                                   full_data[:,-1], test_size=0.2, random_state=1)
    
    
    
    def a_bomb(tr_features, tr_targets, te_features, te_targets, verbose):
        class_nopca = Classification(True, tr_features, tr_targets, 
                                      te_features, te_targets,
                                      PCA = False,
                                      Logit = False,
                                      SVC = False,
                                      NuSVC = True,
                                      SGDC = False,
                                      KNNC=  False,
                                      RNNC = False,
                                      GaussNB = False,
                                      MultiNB = False, 
                                      BernNB = False,
                                      DTC = False,
                                      RFC = False,
                                      ETC = False,
                                      GBC = False,
                                      LDA = False,
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
        pprint(params)
        t0 = time()
        grid_search.fit(inputs, targets)
        print "done in %0.3fs" % (time() - t0)
        print
    
        print "Best CV score: %0.3f" % grid_search.best_score_
        print "Best parameters set:"
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(params.keys()):
            print "\t%s: %r" % (param_name, best_parameters[param_name])
    
    
    ###############################################################################
    
    # Drop the A-bomb
    #a_bomb(train_features, train_targets, test_features, test_targets, True)
    
    ###############################################################################
    
    # Exploring Logistic regression
    
#    pipeline = Pipeline([
#              ('clf', LogisticRegression()),
#              ])
#    
#    parameters = {
#            'clf__C': (np.logspace(-4, 4, 7)),
#            'clf__penalty': ('l1', 'l2'),
#    }
#    
#    workflow(pipeline, parameters, train_features, train_targets)
#    
#    clf = LogisticRegression(penalty='l1', C=1.0)
#    clf.fit(train_features, train_targets)
#    print clf.score(test_features, test_targets)
#    
#    preds = clf.predict(test_features)
#    
#    print(classification_report(test_targets, preds, labels = [1,2,3,4,5,6],target_names=['SB', 'SS', 'ST', 'T', 'TB', 'U']))
#    
#    labels = ['SB', 'SS', 'ST', 'T', 'TB', 'U']
#    cnf = confusion_matrix(test_targets, preds, labels=[1,2,3,4,5,6])
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(cnf)
#    fig.colorbar(cax)
#    ax.set_xticklabels(['']+labels)
#    ax.set_yticklabels(['']+labels)
#    plt.show()
    
    ###############################################################################
    
    # Exploring KNN
    
#    pipeline = Pipeline([
#              ('clf', KNeighborsClassifier()),
#              ])
#    
#    parameters = {
#            'clf__n_neighbors': [3,5],
#            'clf__weights': ['uniform', 'distance'],
#            'clf__algorithm': ['ball_tree', 'kd_tree', 'brute']
#    }
#    
#    workflow(pipeline, parameters, train_features, train_targets)
#    
#    clf = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='ball_tree')
#    clf.fit(train_features, train_targets)
#    print clf.score(test_features, test_targets)
#    
#    preds = clf.predict(test_features)
#    
#    print(classification_report(test_targets, preds, labels = [1,2,3,4],target_names=['B', 'S', 'T', 'U']))
#
#    labels = ['B', 'S', 'T', 'U']
#    cnf = confusion_matrix(test_targets, preds, labels=[1,2,3,4])
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(cnf)
#    fig.colorbar(cax)
#    ax.set_xticklabels(['']+labels)
#    ax.set_yticklabels(['']+labels)
#    plt.show()

    ###############################################################################
    
    # Exploring Random Forests
#    pipeline = Pipeline([
#              ('clf', RandomForestClassifier(n_jobs=1)),
#              ])
#    
#    parameters = {
#            'clf__n_estimators': [5, 10, 15],
#            'clf__criterion': ['entropy', 'gini'],
#            'clf__max_depth': [None, 3, 5, 7],
#            'clf__bootstrap': [True, False]
#    }
#    
#    workflow(pipeline, parameters, train_features, train_targets)
#    
#    clf = RandomForestClassifier(bootstrap=True, n_estimators=15, criterion='entropy', max_depth=5)
#    clf.fit(train_features, train_targets)
#    print clf.score(test_features, test_targets)
#    
#    preds = clf.predict(test_features)
#    
#    print(classification_report(test_targets, preds, labels = [1,2,3,4],target_names=['B', 'S', 'T', 'U']))
#    
#    labels = ['B', 'S', 'T', 'U']
#    cnf = confusion_matrix(test_targets, preds, labels=[1,2,3,4])
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(cnf)
#    fig.colorbar(cax)
#    ax.set_xticklabels(['']+labels)
#    ax.set_yticklabels(['']+labels)
#    plt.show()
    ###############################################################################
    
    # Exploring Gradient Boosting

#    pipeline = Pipeline([
#              ('clf', GradientBoostingClassifier()),
#              ])
#    
#    parameters = {
#            'clf__max_depth': [2, 3, 5],
#            'clf__n_estimators': [200],
#            'clf__learn_rate': [0.05, 0.1, 0.4]
#    }
#    
#    workflow(pipeline, parameters, train_features, train_targets)
#
#    clf = GradientBoostingClassifier(max_depth=5, n_estimators=200, learn_rate=0.4)
#    clf.fit(train_features, train_targets)
#    print clf.score(test_features, test_targets)
#    
#    preds = clf.predict(test_features)
#    
#    print(classification_report(test_targets, preds, labels = [1,2,3,4],target_names=['B', 'S', 'T', 'U']))
#    
#    labels = ['B', 'S', 'T', 'U']
#    cnf = confusion_matrix(test_targets, preds, labels=[1,2,3,4])
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(cnf)
#    fig.colorbar(cax)
#    ax.set_xticklabels(['']+labels)
#    ax.set_yticklabels(['']+labels)
#    plt.show()

    ###############################################################################
    
    # Exploring NuSVC
    
#    pipeline = Pipeline([
#              ('clf', NuSVC()),
#              ])
#    
#    parameters = {
#            'clf__kernel': ['poly', 'rbf'],
#            'clf__nu': [0.05, 0.1, 0.15, 0.25],
#            'clf__degree': [2,3,4],
#            'clf__gamma': [0.0,0.1,1,10]
#    }
    
    #workflow(pipeline, parameters, train_features, train_targets)
    
#    clf = NuSVC(kernel='rbf',nu=0.15)
#    clf.fit(test_features, test_targets)
#    print clf.score(test_features, test_targets)
#    
#    preds = clf.predict(test_features)
#    
#    print(classification_report(test_targets, preds, labels = [1,2,3,4],target_names=['B', 'S', 'T', 'U']))
#    
#    labels = ['B', 'S', 'T', 'U']
#    cnf = confusion_matrix(test_targets, preds, labels=[1,2,3,4])
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    cax = ax.matshow(cnf)
#    fig.colorbar(cax)
#    ax.set_xticklabels(['']+labels)
#    ax.set_yticklabels(['']+labels)
#    plt.show()
    
    ###############################################################################
    
    # Clustering
    
    def clustering(inputs, targets, vis):
        n_samples, n_features = inputs.shape
        n_labels = len(np.unique(targets))
        #labels = digits.target
    
        sample_size = 300
    
        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (n_labels, n_samples, n_features))
        
        print(79 * '_')
        print('% 9s' % 'init'
              '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')
        
        def bench_k_means(estimator, name, data):
            t0 = time()
            estimator.fit(data)
            params = estimator.cluster_centers_
            pprint(params)
            print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
                  % (name, (time() - t0), estimator.inertia_,
                     metrics.homogeneity_score(targets, estimator.labels_),
                     metrics.completeness_score(targets, estimator.labels_),
                     metrics.v_measure_score(targets, estimator.labels_),
                     metrics.adjusted_rand_score(targets, estimator.labels_),
                     metrics.adjusted_mutual_info_score(targets,  estimator.labels_),
                     metrics.silhouette_score(data, estimator.labels_,
                                              metric='euclidean',
                                              sample_size=sample_size)))
        
        bench_k_means(KMeans(init='k-means++', k=n_labels, n_init=10),
                      name="k-means++", data=inputs)
        
        if vis:
            reduced_data = PCA(n_components=2).fit_transform(inputs)
            kmeans = KMeans(init='k-means++', k=n_labels, n_init=10)
            kmeans.fit(reduced_data)
            
            # Step size of the mesh. Decrease to increase the quality of the VQ.
            h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
            
            # Plot the decision boundary. For that, we will asign a color to each
            x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
            y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            # Obtain labels for each point in mesh. Use last trained model.
            Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
            
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            pl.figure(1)
            pl.clf()
            pl.imshow(Z, interpolation='nearest',
                      extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                      cmap=pl.cm.Paired,
                      aspect='auto', origin='lower')
            
            pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
            # Plot the centroids as a white X
            centroids = kmeans.cluster_centers_
            pl.scatter(centroids[:, 0], centroids[:, 1],
                       marker='x', s=169, linewidths=3,
                       color='w', zorder=10)
            pl.title('K-means clustering on the dive data set (PCA-reduced data)\n'
                     'Centroids are marked with white cross')
            pl.xlim(x_min, x_max)
            pl.ylim(y_min, y_max)
            pl.xticks(())
            pl.yticks(())
            pl.show()
        
#    clustering(full_data[:,:-1], full_data[:,-1],False)
    
    est = KMeans(init='k-means++', k=len(np.unique(full_data[:,-1])), n_init=10)
    est.fit(full_data[:,:-1])
    centers = est.cluster_centers_
    
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=3, step=1)
    rfe.fit(full_data[:,:-1], full_data[:,-1])
    ranking = rfe.ranking_
    print ranking
    
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    best_indx = 2
    sec_indx = 9
    third_indx = 16
    
    labels = est.labels_

    fig = matplotlib.figure.Figure(figsize=(6,6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # Set the X Axis label.
    ax.set_xlabel('Median Depth',fontsize=12)

    # Set the Y Axis label.
    ax.set_ylabel('Mean Temp',fontsize=12)
    ax.grid(linestyle='-',color='0.750')#')
    ax.scatter(full_data[:,best_indx],full_data[:,sec_indx],c=full_data[:,-1].astype(np.float))
    
    canvas.print_figure('cluster_scat2.pdf',dpi=500)
    
    