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
    logger = True
    save_fig = False
    number_of_vars_rf = True
    number_of_vars_tree = True
    number_of_vars_rf_low = True
    number_of_vars_tree_low = True
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

    # copy dataframe
    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if number_of_vars_rf:
        # test means for evaluation
        m_acc = []
        m_rec = []
        m_pre = []
        m_auc = []
        # record features left
        n_cols = []

        # set up loop
        for i in range(len(dtfm.columns) - n_variables):
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
            print("Cols left: {}".format(len(fi_dtfm.columns)))
            print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])

            # save results
            m_acc.append(np.mean(accuracy_arr))
            m_rec.append(np.mean(recall_arr))
            m_pre.append(np.mean(precision_arr))
            m_auc.append(np.mean(roc_auc_arr))
            n_cols.append(len(dtfm.columns) - i)

            # print mean and SD
            print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))
        # plot roc_auc for various for various threasholds
        plt.plot(n_cols, m_acc, 'go--')
        plt.plot(n_cols, m_rec, 'bo--')
        plt.plot(n_cols, m_pre, 'co--')
        plt.plot(n_cols, m_auc, 'ro--')
        plt.ylabel('Mean Value')
        plt.xlabel('Number of Variables')
        plt.ylim((0.6, 1))
        plt.legend(['Accuracy', 'Recall', 'Precision', 'ROC'])
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_drop_least_imp')
        else:
            plt.show()


    # cp dataframe
    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if number_of_vars_tree:
        # test means for evaluation
        m_acc = []
        m_rec = []
        m_pre = []
        m_auc = []
        # record features left
        n_cols = []

        # set up loop
        for i in range(len(dtfm.columns) - n_variables):
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
            ]  = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, classifier=DecisionTreeClassifier, logger=logger)

            # remove worst variable from dataframe
            drop_col = fi_dtfm.columns[0]
            print("Cols left: {}".format(len(fi_dtfm.columns)))
            print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])

            # save results
            m_acc.append(np.mean(accuracy_arr))
            m_rec.append(np.mean(recall_arr))
            m_pre.append(np.mean(precision_arr))
            m_auc.append(np.mean(roc_auc_arr))
            n_cols.append(len(dtfm.columns) - i)

            # print mean and SD
            print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))
        # plot roc_auc for various for various threasholds
        plt.plot(n_cols, m_acc, 'go--')
        plt.plot(n_cols, m_rec, 'bo--')
        plt.plot(n_cols, m_pre, 'co--')
        plt.plot(n_cols, m_auc, 'ro--')
        plt.ylabel('Mean Value')
        plt.xlabel('Number of Variables')
        plt.ylim((0.6, 1))
        plt.legend(['Accuracy', 'Recall', 'Precision', 'ROC'])
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_drop_least_imp')
        else:
            plt.show()


    # copy dataframe
    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if number_of_vars_rf_low:
        # test means for evaluation
        m_acc = []
        m_rec = []
        m_pre = []
        m_auc = []
        # record features left
        n_cols = []

        # set up loop
        for i in range(len(dtfm.columns) - n_variables):
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
            drop_col = fi_dtfm.columns[-1]
            print("Cols left: {}".format(len(fi_dtfm.columns)))
            print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])

            # save results
            m_acc.append(np.mean(accuracy_arr))
            m_rec.append(np.mean(recall_arr))
            m_pre.append(np.mean(precision_arr))
            m_auc.append(np.mean(roc_auc_arr))
            n_cols.append(len(dtfm.columns) - i)

            # print mean and SD
            print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))
        # plot roc_auc for various for various threasholds
        plt.plot(n_cols, m_acc, 'go--')
        plt.plot(n_cols, m_rec, 'bo--')
        plt.plot(n_cols, m_pre, 'co--')
        plt.plot(n_cols, m_auc, 'ro--')
        plt.ylabel('Mean Value')
        plt.xlabel('Number of Variables')
        plt.ylim((0.45, 1))
        plt.legend(['Accuracy', 'Recall', 'Precision', 'ROC'])
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_drop_least_imp')
        else:
            plt.show()


    # cp dataframe
    imp_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if number_of_vars_tree_low:
        # test means for evaluation
        m_acc = []
        m_rec = []
        m_pre = []
        m_auc = []
        # record features left
        n_cols = []

        # set up loop
        for i in range(len(dtfm.columns) - n_variables):
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
            ]  = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, classifier=DecisionTreeClassifier, logger=logger)

            # remove worst variable from dataframe
            drop_col = fi_dtfm.columns[-1]
            print("Cols left: {}".format(len(fi_dtfm.columns)))
            print('Dropped Column {}: {}'.format(i,drop_col))
            imp_dtfm = imp_dtfm.drop(columns=[drop_col])

            # save results
            m_acc.append(np.mean(accuracy_arr))
            m_rec.append(np.mean(recall_arr))
            m_pre.append(np.mean(precision_arr))
            m_auc.append(np.mean(roc_auc_arr))
            n_cols.append(len(dtfm.columns) - i)

            # print mean and SD
            print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))
        # plot roc_auc for various for various threasholds
        plt.plot(n_cols, m_acc, 'go--')
        plt.plot(n_cols, m_rec, 'bo--')
        plt.plot(n_cols, m_pre, 'co--')
        plt.plot(n_cols, m_auc, 'ro--')
        plt.ylabel('Mean Value')
        plt.xlabel('Number of Variables')
        plt.ylim((0.45, 1))
        plt.legend(['Accuracy', 'Recall', 'Precision', 'ROC'])
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_drop_least_imp')
        else:
            plt.show()

