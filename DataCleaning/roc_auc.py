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
from evaluate_model import evaluate_model
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def roc_auc(dtfm, labels_col, test_size=0.3, random_state=np.random, logger=False):
    """Returns the roc_auc for an optimised Random Forest
    Trained and tested over a passed in dataset

    USGAE: roc_auc(dtfm, labels_col, test_size=0.3, random_state=np.random, logger=False)

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
    train, test, train_labels, test_labels = train_test_split(dtfm,labels, stratify = labels, test_size = test_size, random_state=random_state)

    #imputation of missing values
    train = train.fillna(train.mean())
    test = test.fillna(test.mean())

    # Hyperparameter grid
    param_grid = {
        'n_estimators': np.linspace(10, 200).astype(int),
        'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
        'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
        'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }
    # Estimator for use in random search
    estimator = RandomForestClassifier(random_state = random_state)
    # Create the random search model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                         scoring = 'roc_auc', cv = 3,
                         n_iter = 10, verbose = 1,
                         random_state=random_state)
    # Fit
    rs.fit(train, train_labels)

    # print result
    if logger:
        print("Best params:\n{}".format(rs.best_params_))

    # Try using the best model
    best_model = rs.best_estimator_

    train_rf_predictions = best_model.predict(train)
    train_rf_probs = best_model.predict_proba(train)[:, 1]

    rf_predictions = best_model.predict(test)
    rf_probs = best_model.predict_proba(test)[:, 1]

    # calculate roc_auc and append to roc_auc_arr
    auc = roc_auc_score(test_labels, rf_probs)
    if logger:
        print("ROC AUC Score=",auc)

    return auc

if __name__ == "__main__":
    # read in dataframe
    dtfm=pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1')
    # remove columns not being used
    dtfm = dtfm.drop(columns=['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA','CELLS_COUNT', 'CLIV'])
    #describe Blast_D8 output vars
    desc_blast = dtfm["BLAST_D8"].describe()
    # create a linspace of 10 points between mu-2sd and mu+2sd
    mean = desc_blast['mean']
    std = desc_blast['std']
    threasholds = np.linspace(mean - 1.5 * std, mean + 1.5 * std, 10)
    print("Threasholds at which we'll calculte ROC_AUC: \n {}".format(threasholds))
    # ROC_AUC array
    roc_auc_arr = []

    #create loop
    for threashold in threasholds:
        print("Optimising a RF for threashold: {}".format(threashold))
        dtfm_temp = dtfm.copy()# make copy of dtfm
        # update blast labels to sit above or below threashold
        dtfm_temp['BLAST_D8'] = dtfm_temp['BLAST_D8'].where(dtfm['BLAST_D8'] >= threashold, other=0)
        dtfm_temp['BLAST_D8'] = dtfm_temp['BLAST_D8'].where(dtfm['BLAST_D8'] < threashold, other=1)

        # run roc_auc func
        score = roc_auc(dtfm_temp, 'BLAST_D8', random_state=50, logger=True)
        roc_auc_arr.append(score)


    # plot roc_auc curve against threashold values
    plt.plot(threasholds, roc_auc_arr)
    plt.ylabel('ROC_AUC')
    plt.xlabel('Threashold')
    plt.ylim(0,1)
    plt.show()
