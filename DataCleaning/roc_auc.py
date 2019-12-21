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

# set a random seed to ensure reproducible runs
RSEED = 50
# read in dataframe
dtfm=pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1')
# remove columns not being used
dtfm = dtfm.drop(columns=['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA','CELLS_COUNT', 'CLIV'])
#describe Blast_D8 output vars
desc_blast = dtfm["BLAST_D8"].describe()
print("Describe Output Vars: \n {}".format(desc_blast))
# create a linspace of 10 points between mu-2sd and mu+2sd
mean = desc_blast['mean']
std = desc_blast['std']
threasholds = np.linspace(mean - 1.5 * std, mean + 1.5 * std, 100)
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
    # print value counts so we can see how split affects data
    print("Blast_D8 value counts:\n {}".format(dtfm_temp['BLAST_D8'].value_counts()))
    labels = np.array(dtfm_temp.pop('BLAST_D8'))#extract labels


    # split data into train and test sets 30% test
    train, test, train_labels, test_labels = train_test_split(dtfm_temp,
            labels, stratify = labels, test_size = 0.3, random_state = RSEED)

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
    estimator = RandomForestClassifier(random_state = RSEED)

    # Create the random search model
    rs = RandomizedSearchCV(estimator, param_grid, n_jobs = -1,
                         scoring = 'roc_auc', cv = 3,
                         n_iter = 10, verbose = 1, random_state=RSEED)

    # Fit
    rs.fit(train, train_labels)

    print("Best params:\n{}".format(rs.best_params_))

    # Try using the best model
    best_model = rs.best_estimator_

    train_rf_predictions = best_model.predict(train)
    train_rf_probs = best_model.predict_proba(train)[:, 1]

    rf_predictions = best_model.predict(test)
    rf_probs = best_model.predict_proba(test)[:, 1]

    # calculate roc_auc and append to roc_auc_arr
    score = roc_auc_score(test_labels, rf_probs)
    roc_auc_arr.append(score)
    print("ROC AUC Score=",score)
    #evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs,test_labels, train_labels, title='Optimised Forest ROC Curve')


# plot roc_auc curve against threashold values
plt.plot(threasholds, roc_auc_arr)
plt.ylabel('ROC_AUC')
plt.xlabel('Threashold')
plt.ylim(0,1)
plt.show()
