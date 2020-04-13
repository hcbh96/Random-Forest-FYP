if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from test_threasholds import test_threasholds

    # define what to run
    logger = True
    save_fig = False
    tree_plot_roc_auc = True
    tree_plot_boxplot = True
    tree_plot_heatmap = True
    rf_plot_roc_auc = True
    rf_plot_boxplot = True
    rf_plot_heatmap = True
    base_case_predictions=True
    n_threasholds = 30


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

    if tree_plot_roc_auc or tree_plot_boxplot or tree_plot_heatmap:
        # test threashold RF
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
        ] = test_threasholds(threasholds, dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

    # plot roc_auc curve against threashold values
    if tree_plot_roc_auc == True:
        plt.plot(threasholds, roc_auc_arr)
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_optimised_threashold')
        else:
            plt.show()

    # plot box plot of most important to least important vars
    if tree_plot_boxplot == True:
        boxplot = fi_dtfm.drop('mean').boxplot(rot=90)
        plt.ylabel('Blastocyst Threashold')
        plt.xlabel('Feature')
        if save_fig:
            plt.savefig('var_imp_box_whisker')
        else:
            plt.show()

    # plot heatmap of most to least important vars
    if tree_plot_heatmap:
        sns.heatmap(fi_dtfm)
        plt.xlabel('Feature')
        plt.ylabel('Blastocyt Threashold')
        if save_fig:
            plt.savefig('var_imp_heatmap')
        else:
            plt.show()

    if rf_plot_roc_auc or rf_plot_boxplot or rf_plot_heatmap:
        # test threashold RF
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
        ] = test_threasholds(threasholds, dtfm, dep_key='BLAST_D8', random_state=50, logger=logger)

    # plot roc_auc curve against threashold values
    if rf_plot_roc_auc == True:
        plt.plot(threasholds, roc_auc_arr)
        plt.ylabel('ROC_AUC')
        plt.xlabel('Threashold')
        plt.ylim(0,1)
        plt.tight_layout()
        if save_fig:
            plt.savefig('roc_auc_optimised_threashold')
        else:
            plt.show()

    # plot box plot of most important to least important vars
    if rf_plot_boxplot == True:
        boxplot = fi_dtfm.drop('mean').boxplot(rot=90)
        plt.ylabel('Blastocyst Threashold')
        plt.xlabel('Feature')
        if save_fig:
            plt.savefig('var_imp_box_whisker')
        else:
            plt.show()

    # plot heatmap of most to least important vars
    if rf_plot_heatmap:
        sns.heatmap(fi_dtfm)
        plt.xlabel('Feature')
        plt.ylabel('Blastocyt Threashold')
        if save_fig:
            plt.savefig('var_imp_heatmap')
        else:
            plt.show()


    if base_case_predictions:
        # this can be used to ensure that on a random output the classifier does not work
        # therefore acting as a proof that the results are real
        dtfm['randNumCol'] = np.random.randint(1, 100, dtfm.shape[0])
        dtfm = dtfm.drop(columns=['BLAST_D8'])
        # test threashold RF
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
        ] = test_threasholds(threasholds, dtfm, classifier=DecisionTreeClassifier, dep_key='randNumCol', random_state=50, logger=logger)


        # test threashold RF
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
        ] = test_threasholds(threasholds, dtfm, dep_key='randNumCol', random_state=50, logger=logger)


        # print mean and SD
        if logger:
            print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))

        # print mean and SD
        if logger:
            print("Accuracy Mean: {}, SD: {}".format(np.mean(c_accuracy_arr), np.std(c_accuracy_arr)))
            print("Recall Mean: {}, SD: {}".format(np.mean(c_recall_arr), np.std(c_recall_arr)))
            print("Precision Mean: {}, SD: {}".format(np.mean(c_precision_arr), np.std(c_precision_arr)))
            print("ROC AUC Mean: {}, SD: {}".format(np.mean(c_roc_auc_arr),np.std(c_roc_auc_arr)))

        # plot roc_auc for various for various threasholds
        d = {
                "RF":roc_auc_arr,
                "CART": c_roc_auc_arr,
            }

        if logger:
            print("Metrics:\n{}".format(d))

        plot_dtfm = pd.DataFrame(data=d)
        plot_dtfm = plot_dtfm.boxplot()
        plt.ylabel('Method')
        plt.xlabel('AUC')
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_fig:
            plt.savefig('classifiers_on_random_set')
        else:
            plt.show()
