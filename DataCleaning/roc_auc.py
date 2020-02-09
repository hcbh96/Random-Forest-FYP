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

def roc_auc(dtfm, labels_col, test_size=0.3, random_state=np.random, logger=False):
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
    print(dtfm.head())

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

    # calculate variables of most importance in model
    fi_model = pd.DataFrame({'feature': features,
                'importance': best_model.feature_importances_})

    if logger:
        print("Features importance in RF:\n{}".format(fi_model.sort_values('importance', 0, ascending=True)))

    return [auc, fi_model]


def test_threasholds(threasholds, dtfm, dep_key='BLAST_D8', random_state=50, logger=False):
    if logger== True:
        print("Threasholds at which we'll calculte ROC_AUC: \n {}".format(threasholds))

    # ROC_AUC array
    roc_auc_arr = []
    fi_dtfm = pd.DataFrame()
    dtfm_index = 0

    #create loop
    for threashold in threasholds:
        if logger == True:
            print("Optimising a RF for threashold: {}".format(threashold))
        dtfm_temp = dtfm.copy()# make copy of dtfm
        # update blast labels to sit above or below threashold
        dtfm_temp[dep_key] = dtfm_temp[dep_key].where(dtfm[dep_key] >= threashold, other=0)
        dtfm_temp[dep_key] = dtfm_temp[dep_key].where(dtfm[dep_key] < threashold, other=1)

        # run roc_auc func
        [score, fi] = roc_auc(dtfm_temp, dep_key, random_state=random_state, logger=logger)

        # create first index
        if dtfm_index == 0:
            fi_dtfm.insert(0, 'feature', fi.feature)

        fi_dtfm.insert(dtfm_index + 1, round(threashold, 2),fi.importance)
        dtfm_index += 1
        roc_auc_arr.append(score)

    # redefine index
    fi_dtfm = fi_dtfm.set_index('feature')
    fi_dtfm['mean'] = fi_dtfm.mean(axis=1)
    fi_dtfm = fi_dtfm.sort_values('mean', 0, ascending=True)
    fi_dtfm = fi_dtfm.transpose()

    return [
            roc_auc_arr,
            fi_dtfm,
           ]




if __name__ == "__main__":
    # define what to run
    logger = True
    plot_roc_auc = True
    plot_boxplot = False
    plot_heatmap = False
    high_threasholds = False
    low_threasholds = False
    m_v_i = False
    n_threasholds = 10

    #cols to drop
    discrete_cols = ['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA',   'CELLS_COUNT', 'CLIV']
    discrete_cols = discrete_cols + ['SUB_1_RP','SUB_2_H','SUB_3_LS','SUB_4_LP']
    motility_cols = ['VAP','VSL','VCL','ALH','BCF','STR','LIN','MOTILE_PCT',
'PROGRESSIVE_PCT','RAPID_PCT','MEDIUM_PCT','SLOW_PCT','STATIC_PCT']
    integrity_cols = ['AI','PI','ALTO','FRAG_CRO',
'MOT_PRE','MOT_POS','CONC_CAMARA','VF','AD']

    # read in dataframe
    dtfm=pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1', index_col=0)

    # remove columns not being used
    dtfm = dtfm.drop(columns=discrete_cols)

    # create a linspace of 10 points between mu-2sd and mu+2sd
    desc_blast = dtfm["BLAST_D8"].describe()
    mean = desc_blast['mean']
    std = desc_blast['std']
    threasholds = np.linspace(mean - 1.5 * std, mean + 1.5 * std, n_threasholds)

    # test threashold norm
    [roc_auc_arr, fi_dtfm] = test_threasholds(threasholds, dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

    # plot roc_auc curve against threashold values
    if plot_roc_auc == True:
        plt.plot(threasholds, roc_auc_arr)
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.show()

    # plot box plot of most important to least important vars
    if plot_boxplot == True:
        boxplot = fi_dtfm.drop('mean').boxplot(rot=90)
        plt.ylabel('Blastocyst Threashold')
        plt.xlabel('Feature')
        plt.tight_layout()
        plt.show()

    # plot heatmap of most to least important vars
    if plot_heatmap:
        sns.heatmap(fi_dtfm)
        plt.xlabel('Feature')
        plt.ylabel('Blastocyt Threashold')
        plt.tight_layout()
        plt.show()

    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if high_threasholds == True:
        drop_cols = ['Basis']
        # set up loop
        for i in range(len(dtfm.columns) - 1):
            RED = i/(len(dtfm.columns) + 1)
            GREEN = 0
            BLUE = 1 - i/(len(dtfm.columns) + 1)
            print("colors RED:{}, GREEN:{}, BLUE:{}".format(RED, GREEN, BLUE))
            # run roc_auc
            [roc_auc_arr, fi_dtfm] = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

            # remove worst variable from dataframe
            drop_col = fi_dtfm.columns[0]
            drop_cols.append('{}:{}'.format(i,drop_col))
            print("Cols left: {}".format(len(fi_dtfm.columns)))
            print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])
            # save roc_auc
            plt.plot(threasholds, roc_auc_arr, color=[RED, GREEN, BLUE])

        # plot roc_auc for various for various threasholds
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.legend(drop_cols, ncol=4)
        plt.show()


    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if low_threasholds == True:
        drop_cols = ['Basis']
        # set up loop
        for i in range(len(dtfm.columns) - 1):
            RED = i/(len(dtfm.columns)+1)
            GREEN = 0
            BLUE = 1 - i/(len(dtfm.columns) + 1)
            print("colors RED:{}, GREEN:{}, BLUE:{}".format(RED, GREEN, BLUE))
            # run roc_auc
            [roc_auc_arr, fi_dtfm] = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

            # remove worst variable from dataframe
            drop_col = fi_dtfm.columns[-1]
            drop_cols.append('{}:{}'.format(i,drop_col))
            print("Cols left: {}".format(len(fi_dtfm.columns)))
            print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])
            # save roc_auc
            plt.plot(threasholds, roc_auc_arr, color=[RED, GREEN, BLUE])

        # plot roc_auc for various for various threasholds
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.legend(drop_cols, ncol=4)
        plt.show()


    if m_v_i == True:
        # plot for integrity vs motility
        integrity_dtfm = dtfm.drop(columns=motility_cols)
        motility_dtfm = dtfm.drop(columns=integrity_cols)

        # test integrity
        [roc_auc_arr, fi_dtfm] = test_threasholds(threasholds, integrity_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)
        plt.plot(threasholds, roc_auc_arr)

        [roc_auc_arr, fi_dtfm] = test_threasholds(threasholds, motility_dtfm,dep_key='BLAST_D8', random_state=50, logger=logger)
        plt.plot(threasholds, roc_auc_arr)

        #plot integrity vs motility
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.legend(['Integrity','Motility'])
        plt.show()
