"""
In this file I will look to do the following

1. Loop thought various threasholds that separate top lowest 90% of values
    a. Optimise a RF for the threashold
    b. Run the RF over the test data
    c. record the ROC_AUC for the RF on the test data
2. Plot the ROC_AUC vs the threashold"""

import pandas as pd
import numpy as np

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
threasholds = np.linspace(mean - 1.5 * std, mean + 1.5 * std, 10)
print("Threasholds at which we'll calculte ROC_AUC: \n {}".format(threasholds))

#create loop
for threashold in threasholds:
    print("Optimising a RF for threashold: {}".format(threashold))
    dtfm_temp = dtfm.copy# make copy of dtfm
    # update blast labels to sit above or below threashold
    dtfm_temp['BLAST_D8'] = dtfm_temp['BLAST_D8'].where(dtfm['BLAST_D8'] >= threashold, other=0)
    dtfm_temp['BLAST_D8'] = dtfm_temp['BLAST_D8'].where(dtfm['BLAST_D8'] < threashold, other=1)
    # print value counts so we can see how split affects data
    print("Blast_D8 value counts:\n {}".format(dtfm_temp['BLAST_D8'].value_counts()))
    labels = np.array(dtfm_B.pop('BLAST_D8'))#extract labels


    # split data into train and test sets 30% test
    train, test, train_labels, test_labels = train_test_split(dtfm_B,
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
