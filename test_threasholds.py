"""
In this file I will look to do the following

1. Loop thought various threasholds that separate top lowest 90% of values
    a. Optimise a RF for the threashold
    b. Run the RF over the test data
    c. record the ROC_AUC for the RF on the test data
2. Plot the ROC_AUC vs the threashold"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from evaluate_model import evaluate_model

def find_best_params(dtfm, labels_col, classifier=RandomForestClassifier,
        test_size=0.3, random_state=np.random, logger=False):
    """Returns the best params in terms of roc_auc for an optimised Random Forest
    Trained and tested over a passed in dataset

    USGAE: find_best_params(dtfm, labels_col, test_size=0.3, random_state=np.random, logger=logger)

    INPUTS:
        dtfm: dataframe to train and test model on
        labels_col: column title containing the classification flag

    **Optional** (default)
        test_size: (0.3) proportion of data set to use for testing
        random_state: (np.random) seed used by the random number generator
        logger: (False) wheather to log outputs or not


    OUTPUT:
        roc_auc: (scalar) Area under the receiver operating
            charachteristic curve
    """

    dtfm_labels = dtfm.pop(labels_col)
    # separate the labels from the data
    labels = np.array(dtfm_labels)
    print(dtfm.head())

    # print value counts so we can see how split affects data
    if logger:
        print("Output value count:\n {}".format(dtfm_labels.value_counts()))

    # split data into train and test sets split% test
    train, test, train_labels, test_labels = train_test_split(dtfm, labels, stratify = labels, test_size = test_size, random_state=random_state)

    #imputation of missing values
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())

    c_name = classifier.__name__
    if c_name == 'RandomForestClassifier':
        # Hyperparameter grid
        param_grid = {
            'n_estimators': np.linspace(10, 200, 10).astype(int),
            'max_depth': [None] + list(np.linspace(3, 20, 10).astype(int)),
            'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
            'max_leaf_nodes': [None] + list(np.linspace(10, 50, 10).astype(int)),
            'min_samples_split': [2],
            'bootstrap': [True],
            }
    elif c_name == 'DecisionTreeClassifier':
        # Hyperparameter grid
        param_grid = {
            'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
            'max_leaf_nodes': [None] + list(np.linspace(10, 50, 10).astype(int)),
            'min_samples_split': [2],
        }
    else:
        raise ValueError('That is not a supported Classifier')

    # Estimator for use in random search
    estimator = classifier(random_state = random_state)
    # Create the random search model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                         scoring = 'roc_auc', cv = 3,
                         n_iter = 10, verbose = 1,
                         random_state=random_state)
    # Fit
    rs.fit(train, train_labels)

    print("Best params:\n{}".format(rs.best_params_))
    # print result
    if logger:
        print("Best params:\n{}".format(rs.best_params_))

    return rs.best_params_


def roc_auc(dtfm, labels_col, c_params, classifier=RandomForestClassifier,
        test_size=0.3, random_state=np.random, logger=False, optimise=True):
    """Returns the roc_auc for an optimised Random Forest
    Trained and tested over a passed in dataset

    USGAE: roc_auc(dtfm, labels_col, test_size=0.3, random_state=np.random, logger=logger)

    INPUTS:
        dtfm: dataframe to train and test model on
        labels_col: column title containing the classification flag

    **Optional** (default)
        test_size: (0.3) proportion of data set to use for testing
        random_state: (np.random) seed used by the random number generator
        logger: (False) wheather to log outputs or not


    OUTPUT:
        roc_auc: (scalar) Area under the receiver operating
            charachteristic curve
    """

    dtfm_labels = dtfm.pop(labels_col)
    # separate the labels from the data
    labels = np.array(dtfm_labels)

    # print value counts so we can see how split affects data
    if logger:
        print("Output value count:\n {}".format(dtfm_labels.value_counts()))

    # split data into train and test sets split% test
    train, test, train_labels, test_labels = train_test_split(dtfm, labels, stratify = labels, test_size = test_size, random_state=random_state)

    #imputation of missing values
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())

    # Features for feature importances
    features = list(train.columns)
    c_name = classifier.__name__
    if c_name == 'RandomForestClassifier':
        # tune model
        model = classifier(n_estimators=c_params['n_estimators'], max_depth=c_params['max_depth'],
                max_features=c_params['max_features'], max_leaf_nodes=c_params['max_leaf_nodes'],
                min_samples_split=c_params['min_samples_split'], bootstrap=c_params['bootstrap'],
                random_state = random_state)
    elif c_name == 'DecisionTreeClassifier':
        # tune model
        model = classifier(max_features=c_params['max_features'], max_leaf_nodes=c_params['max_leaf_nodes'],
                min_samples_split=c_params['min_samples_split'], random_state=random_state)
    else:
        raise ValueError('That is not a supported Classifier')

    # Fit
    model.fit(train, train_labels)

    train_predictions = model.predict(train)
    train_probs = model.predict_proba(train)[:, 1]

    predictions = model.predict(test)
    probs = model.predict_proba(test)[:, 1]

    [baseline, results, train_results ] = evaluate_model(predictions, probs,
            train_predictions, train_probs,
            test_labels, train_labels, logger=logger)

    # calculate variables of most importance in model
    fi_model = pd.DataFrame({'feature': features,
                'importance': model.feature_importances_})

    if logger:
        print("Features importance in RF:\n{}".format(fi_model.sort_values('importance', 0, ascending=True)))

    return [
            baseline, results,
            train_results, fi_model,
            ]

def test_threasholds(threasholds, dtfm, classifier=RandomForestClassifier,
        dep_key='BLAST_D8', random_state=50, logger=False, optimise=True):
    if logger== True:
        print("Threasholds at which we'll calculte ROC_AUC: \n {}".format(threasholds))


    # ROC_AUC array
    roc_auc_arr = []
    roc_auc_train_arr = []
    # Precision
    precision_arr = []
    precision_train_arr = []
    # Recall
    recall_arr = []
    recall_train_arr = []
    #Accuracy
    accuracy_arr = []
    accuracy_train_arr = []
    # feature importance
    fi_dtfm = pd.DataFrame()
    dtfm_index = 0
    # set base for c_params
    c_params = {'n_estimators': 94, 'min_samples_split': 2, 'max_leaf_nodes': 36, 'max_features': 0.5, 'max_depth': 18, 'bootstrap': True}
    #create loop
    for threashold in threasholds:
        if logger == True:
            print("Optimising a RF for threashold: {}".format(threashold))
        dtfm_temp = dtfm.copy()# make copy of dtfm
        # update blast labels to sit above or below threashold
        dtfm_temp[dep_key] = dtfm_temp[dep_key].where(dtfm[dep_key] >= threashold, other=0)
        dtfm_temp[dep_key] = dtfm_temp[dep_key].where(dtfm[dep_key] < threashold, other=1)

        # bool trigger for multiple optimisations
        if optimise:
            dtfm_optimise = dtfm_temp.copy()
            c_params = find_best_params(dtfm_optimise, dep_key, classifier=classifier, random_state=random_state, logger=logger)

        # maybe run optimise once then use params from optimise for subsequent threasholds
        [
            baseline, results,
            train_results, fi_model,
        ] = roc_auc(dtfm_temp, dep_key, c_params, classifier=classifier,
                random_state=random_state, logger=logger, optimise=optimise)

        # create first index
        if dtfm_index == 0:
            fi_dtfm.insert(0, 'feature', fi_model.feature)

        fi_dtfm.insert(dtfm_index + 1, round(threashold, 2),fi_model.importance)
        dtfm_index += 1

        # ROC_AUC
        roc_auc_arr.append(results['roc'])
        roc_auc_train_arr.append(train_results['roc'])

        # Precision
        precision_arr.append(results['precision'])
        precision_train_arr.append(train_results['precision'])

        # Recall
        recall_arr.append(results['recall'])
        recall_train_arr.append(train_results['recall'])

        # Precision
        accuracy_arr.append(results['accuracy'])
        accuracy_train_arr.append(train_results['accuracy'])

    # redefine index
    fi_dtfm = fi_dtfm.set_index('feature')
    fi_dtfm['mean'] = fi_dtfm.mean(axis=1)
    fi_dtfm = fi_dtfm.sort_values('mean', 0, ascending=True)
    fi_dtfm = fi_dtfm.transpose()

    return [
            fi_dtfm,
            roc_auc_arr,
            roc_auc_train_arr,
            precision_arr,
            precision_train_arr,
            recall_arr,
            recall_train_arr,
            accuracy_arr,
            accuracy_train_arr,
           ]


