"""
This file will be used to assess
how reducing the variable set will
affect the classification measurments
"""

if __name__ == '__main__':
    from test_threasholds import test_threasholds
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier

    # define what to run
    logger = False
    save_fig = False
    rf_drop_var_plot = True
    tree_drop_var_plot = True
    n_threasholds = 30
    n_variables = 1


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

    # copy original dataframe
    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if rf_drop_var_plot:
        drop_cols = ['Basis']
        # set up loop
        for i in range(len(dtfm.columns) - n_variables):
            RED = i/(len(dtfm.columns)+1)
            GREEN = 0
            BLUE = 1 - i/(len(dtfm.columns) + 1)
            print("colors RED:{}, GREEN:{}, BLUE:{}".format(RED, GREEN, BLUE))
            # run roc_auc
            [
                fi_dtfm,
                roc_auc_arr,
                roc_auc_train_arr,
                precision_arr,
                precision_train_arr,
                recall_arr,
                recall_train_arr,
                accuracy_arr,
                accuracy_train_arr,
            ]  = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

            # remove worst variable from dataframe
            drop_col = fi_dtfm.columns[0]
            drop_cols.append('{}:{}'.format(i,drop_col))
            if logger:
                print("Cols left: {}".format(len(fi_dtfm.columns)))
                print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])
            # save roc_auc
            plt.plot(threasholds, roc_auc_arr, color=[RED, GREEN, BLUE])

        # plot roc_auc for various for various threasholds
        if logger:
            print("Dropped Cols: {}".format(drop_cols))
            print("Remaining Cols: {}".format(fi_dtfm.columns))
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.legend(drop_cols, ncol=4)
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_drop_least_imp')
        else:
            plt.show()


    # copy original dataframe
    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if tree_drop_var_plot:
        drop_cols = ['Basis']
        # set up loop
        for i in range(len(dtfm.columns) - n_variables):
            RED = i/(len(dtfm.columns)+1)
            GREEN = 0
            BLUE = 1 - i/(len(dtfm.columns) + 1)
            if logger:
                print("colors RED:{}, GREEN:{}, BLUE:{}".format(RED, GREEN, BLUE))
            # run roc_auc
            [
                fi_dtfm,
                roc_auc_arr,
                roc_auc_train_arr,
                precision_arr,
                precision_train_arr,
                recall_arr,
                recall_train_arr,
                accuracy_arr,
                accuracy_train_arr,
            ]  = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger, classifier=DecisionTreeClassifier)

            # remove worst variable from dataframe
            drop_col = fi_dtfm.columns[0]
            drop_cols.append('{}:{}'.format(i,drop_col))
            if logger:
                print("Cols left: {}".format(len(fi_dtfm.columns)))
                print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])
            # save roc_auc
            plt.plot(threasholds, roc_auc_arr, color=[RED, GREEN, BLUE])

        # plot roc_auc for various for various threasholds
        if logger:
            print("Dropped Cols: {}".format(drop_cols))
            print("Remaining Cols: {}".format(fi_dtfm.columns))
        plt.ylabel('AUC ROC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.legend(drop_cols, ncol=4)
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_drop_least_imp')
        else:
            plt.show()
