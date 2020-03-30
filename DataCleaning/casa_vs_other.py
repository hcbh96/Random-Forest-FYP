"""
This file will be used to measure the classifier with full dataset
and the classifer comparison between casa and other
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
    logger = True
    save_fig = False
    full_set = True
    casa_other = True
    n_threasholds = 30
    n_variables = 1


    #cols to drop
    discrete_cols = ['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA',   'CELLS_COUNT', 'CLIV']
    discrete_cols = discrete_cols + ['SUB_1_RP','SUB_2_H','SUB_3_LS','SUB_4_LP']
    casa_cols = ['VAP', 'VSL', 'VCL', 'ALH', 'BCF', 'STR', 'LIN', 'MOTILE_PCT', 'PROGRESSIVE_PCT', 'RAPID_PCT', 'MEDIUM_PCT', 'SLOW_PCT', 'STATIC_PCT']
    other_cols = ['AI', 'PI', 'ALTO', 'FRAG_CRO', 'MOT_PRE', 'MOT_POS', 'CONC_CAMARA', 'VF', 'AD']
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
    if full_set:

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
        ] = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

        # print mean and SD
        if logger:
            print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))
        # plot roc_auc for various for various threasholds
        d = {
                "Accuracy":accuracy_arr,
                "Recall":recall_arr,
                "Precision":precision_arr,
                "AUC":roc_auc_arr,
            }
        plot_dtfm = pd.DataFrame(data=d)
        plot_dtfm = plot_dtfm.boxplot()
        plt.ylabel('Measure')
        plt.xlabel('Value')
        plt.tight_layout()
        if save_fig:
            plt.savefig('full_data_set_measures')
        else:
            plt.show()


    # copy dataframe
    casa_dtfm = dtfm.drop(columns=other_cols)
    other_dtfm = dtfm.drop(columns=casa_cols)
    # run roc_auc multiple times
    if casa_other:

        # run roc_auc
        [
            c_fi_dtfm,
            c_roc_auc_arr,
            c_roc_auc_train_arr,
            c_precision_arr,
            c_precision_train_arr,
            c_recall_arr,
            c_recall_train_arr,
            c_accuracy_arr,
            c_accuracy_train_arr,
        ] = test_threasholds(threasholds, casa_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

        # run roc_auc
        [
            o_fi_dtfm,
            o_roc_auc_arr,
            o_roc_auc_train_arr,
            o_precision_arr,
            o_precision_train_arr,
            o_recall_arr,
            o_recall_train_arr,
            o_accuracy_arr,
            o_accuracy_train_arr,
        ] = test_threasholds(threasholds, other_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

        # print mean and SD
        if logger:
            print("Accuracy Mean: {}, SD: {}".format(np.mean(c_accuracy_arr), np.std(c_accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(c_recall_arr), np.std(c_recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(c_precision_arr), np.std(c_precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(c_roc_auc_arr),np.std(c_roc_auc_arr)))
            print("Accuracy Mean: {}, SD: {}".format(np.mean(o_accuracy_arr), np.std(o_accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(o_recall_arr), np.std(o_recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(o_precision_arr), np.std(o_precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(o_roc_auc_arr),np.std(o_roc_auc_arr)))
        # plot roc_auc for various for various threasholds
        d = {
                "CASA Accuracy":c_accuracy_arr,
                "Other Accuracy":o_accuracy_arr,
                "Accuracy":accuracy_arr,
                "CASA Recall":c_recall_arr,
                "Other Recall":o_recall_arr,
                "Recall":recall_arr,
                "CASA Precision":c_precision_arr,
                "Other Precision":o_precision_arr,
                "Precision":precision_arr,
                "CASA AUC":c_roc_auc_arr,
                "Other AUC":o_roc_auc_arr,
                "AUC":roc_auc_arr,
            }
        if logger:
            print("Metrics:\n{}".format(d))

        plot_dtfm = pd.DataFrame(data=d)
        plot_dtfm = plot_dtfm.boxplot()
        plt.ylabel('Measure')
        plt.xlabel('Value')
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_fig:
            plt.savefig('casa_other_data_set_measures')
        else:
            plt.show()


