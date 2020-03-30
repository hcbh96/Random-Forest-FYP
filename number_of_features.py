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
    from tqdm import tqdm

    # define what to run
    logger = False
    save_fig = False
    number_of_vars_rf = True
    number_of_vars_tree = False
    number_of_vars_rf_low = False
    number_of_vars_tree_low = False
    roc_comp = False
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
    threasholds = np.linspace(mean - 1 * std, mean + 1 * std, n_threasholds)

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
        for i in tqdm(range(len(dtfm.columns) - n_variables)):
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
            if logger:
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
            if logger:
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
        for i in tqdm(range(len(dtfm.columns) - n_variables)):
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
            if logger:
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
            if logger:
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
        for i in tqdm(range(len(dtfm.columns) - n_variables)):
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
            if logger:
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
            if logger:
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
        for i in tqdm(range(len(dtfm.columns) - n_variables)):
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
            if logger:
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
            if logger:
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

    # copy dataframe
    imp1_dtfm = dtfm.copy()
    imp2_dtfm = dtfm.copy()
    # run roc_auc multiple times
    if roc_comp:
        # test means for evaluation
        m1_acc = []
        m1_rec = []
        m1_pre = []
        m1_auc = []
        # record features left
        n1_cols = []

        # test means for evaluation
        m2_acc = []
        m2_rec = []
        m2_pre = []
        m2_auc = []
        # record features left
        n2_cols = []

        # set up loop
        for i in tqdm(range(len(dtfm.columns) - n_variables)):
            # run roc_auc
            [
                fi1_dtfm,
                roc1_auc_arr,
                roc1_auc_train_arr,
                precision1_arr,
                precision1_train_arr,
                recall1_arr,
                recall1_train_arr,
                accuracy1_arr,
                accuracy1_train_arr,
            ]  = test_threasholds(threasholds, imp1_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)
            # run roc_auc
            [
                fi2_dtfm,
                roc2_auc_arr,
                roc2_auc_train_arr,
                precision2_arr,
                precision2_train_arr,
                recall2_arr,
                recall2_train_arr,
                accuracy2_arr,
                accuracy2_train_arr,
            ]  = test_threasholds(threasholds, imp2_dtfm, dep_key='BLAST_D8',random_state=50, logger=logger)

            # remove worst variable from dataframe
            drop1_col = fi1_dtfm.columns[0]
            if logger:
                print("Cols left: {}".format(len(fi1_dtfm.columns)))
                print('Dropped Column {}: {}'.format(i,drop1_col))
            imp1_dtfm = imp1_dtfm.drop(columns=[drop1_col])

            # remove best variable from dataframe
            drop2_col = fi2_dtfm.columns[-1]
            if logger:
                print("Cols left: {}".format(len(fi2_dtfm.columns)))
                print('Dropped Column {}: {}'.format(i,drop2_col))
            imp2_dtfm = imp2_dtfm.drop(columns=[drop2_col])

            # save results
            m1_acc.append(np.mean(accuracy1_arr))
            m1_rec.append(np.mean(recall1_arr))
            m1_pre.append(np.mean(precision1_arr))
            m1_auc.append(np.mean(roc1_auc_arr))
            n1_cols.append(len(dtfm.columns) - i)

            # save results
            m2_acc.append(np.mean(accuracy2_arr))
            m2_rec.append(np.mean(recall2_arr))
            m2_pre.append(np.mean(precision2_arr))
            m2_auc.append(np.mean(roc2_auc_arr))
            n2_cols.append(len(dtfm.columns) - i)

            # print mean and SD
            if logger:
                print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy1_arr), np.std(accuracy1_arr)))
                print("Recall Mean: {}, SD: {}".format(np.mean(recall1_arr), np.std(recall1_arr)))
                print("Precision Mean: {}, SD: {}".format(np.mean(precision1_arr), np.std(precision1_arr)))
                print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc1_auc_arr),np.std(roc1_auc_arr)))

            # print mean and SD
            if logger:
                print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy2_arr), np.std(accuracy2_arr)))
                print("Recall Mean: {}, SD: {}".format(np.mean(recall2_arr), np.std(recall2_arr)))
                print("Precision Mean: {}, SD: {}".format(np.mean(precision2_arr), np.std(precision2_arr)))
                print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc2_auc_arr),np.std(roc2_auc_arr)))

        # plot roc_auc for various for various threasholds
        plt.plot(n1_cols, m1_auc, 'ro--')
        plt.plot(n2_cols, m2_auc, 'bo--')
        plt.ylabel('Mean Value')
        plt.xlabel('Number of Variables')
        plt.ylim((0.6, 1))
        plt.legend(['Most important', 'Least important'])
        plt.tight_layout()
        if save_fig:
            plt.savefig('rf_roc_low_hig_comp')
        else:
            plt.show()

