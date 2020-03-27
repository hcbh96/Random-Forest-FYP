"""
This file will be used to assess
all permutations of the clasifier
when using only 7 variables
"""

if __name__ == '__main__':
    from test_threasholds import test_threasholds
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    import itertools
    from tqdm import tqdm
    import seaborn as sns

    # define what to run
    logger = False
    save_fig = False
    seven_permutations = True
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
    blast_d8 = dtfm["BLAST_D8"]
    desc_blast = blast_d8.describe()
    mean = desc_blast['mean']
    std = desc_blast['std']
    threasholds = [mean]

    #remove blast_d8 from dtfm (N.B it will be added in later)
    dtfm = dtfm.drop(columns=["BLAST_D8"])

    # make permutations containing 7 vars
    total_cols = dtfm.columns
    if logger:
        print("dtfm num cols: {}".format(len(total_cols)))

    if seven_permutations:
        def find_subsets(s,n):
            return list(itertools.combinations(s,n))

        subsets = find_subsets(total_cols, 7)
        if logger:
            print("Number of subsets: {}".format(len(subsets)))

        # test means for evaluation
        m_acc = []
        m_rec = []
        m_pre = []
        m_auc = []
        best_permutations = pd.DataFrame({
                    "acc": (),
                    "rec": (),
                    "pre": (),
                    "auc": (),
                })
        best_acc = 0
        best_rec = 0
        best_pre = 0
        best_auc = 0
        worst_permutations = pd.DataFrame({
                    "acc": (),
                    "rec": (),
                    "pre": (),
                    "auc": (),
                })
        worst_acc = 1
        worst_rec = 1
        worst_pre = 1
        worst_auc = 1

        # number of exel rows in each file
        file_number = 0
        save_loop = 17000
        save_dtfm = pd.DataFrame()

        save_count = 0

        # set up loop and create progress bar
        for subset in tqdm(subsets):
            save_loop = save_loop - 1
            if save_loop <= 0:
                save_loop = 17000
                save_count = save_count + 1
                # write to execel empty save_dtfm
                save_dtfm.to_excel('excel_permutations/permutations_{}.xlsx'.format(file_number))
                file_number = file_number + 1
                save_dtfm = pd.DataFrame()

            use_cols = np.asarray(subset)
            imp_dtfm = dtfm[use_cols]

            if logger:
                print("Use Cols: {}".format(use_cols))

            # add blast_d8 back into dataframe
            imp_dtfm['BLAST_D8'] = blast_d8

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
            ]  = test_threasholds(threasholds, imp_dtfm, dep_key='BLAST_D8', random_state=50, logger=logger, optimise=False)

            # calc metrics
            acc_mean = np.mean(accuracy_arr)
            rec_mean = np.mean(recall_arr)
            pre_mean = np.mean(precision_arr)
            auc_mean = np.mean(roc_auc_arr)
            # save results
            m_acc.append(acc_mean)
            m_rec.append(rec_mean)
            m_pre.append(pre_mean)
            m_auc.append(auc_mean)

            # save best permutation
            if acc_mean > best_acc:
                best_acc = acc_mean
                best_permutations["acc"] = subset
            if rec_mean > best_rec:
                best_rec = rec_mean
                best_permutations["rec"] = subset
            if pre_mean > best_pre:
                best_pre = pre_mean
                best_permutations["pre"] = subset
            if auc_mean > best_auc:
                best_auc = auc_mean
                best_permutations["auc"] = subset
            # save worst permustation
            if acc_mean < worst_acc:
                worst_acc = acc_mean
                worst_permutations["acc"] = subset
            if rec_mean < worst_rec:
                worst_rec = rec_mean
                worst_permutations["rec"] = subset
            if pre_mean < worst_pre:
                worst_pre = pre_mean
                worst_permutations["pre"] = subset
            if auc_mean < worst_auc:
                worst_auc = auc_mean
                worst_permutations["auc"] = subset

            save_dtfm = save_dtfm.append({
                    "Accuracy": acc_mean,
                    "Recall": rec_mean,
                    "Precision": pre_mean,
                    "AUC": auc_mean,
                    "Var 0": use_cols[0],
                    "Var 1": use_cols[1],
                    "Var 2": use_cols[2],
                    "Var 3": use_cols[3],
                    "Var 4": use_cols[4],
                    "Var 5": use_cols[5],
                    "Var 6": use_cols[6],
                },ignore_index=True)

            # print mean and SD
            if logger:
                print("Accuracy Mean: {}, SD: {}".format(np.mean(accuracy_arr), np.std(accuracy_arr)))
                print("Recall Mean: {}, SD: {}".format(np.mean(recall_arr), np.std(recall_arr)))
                print("Precision Mean: {}, SD: {}".format(np.mean(precision_arr), np.std(precision_arr)))
                print("ROC AUC Mean: {}, SD: {}".format(np.mean(roc_auc_arr),np.std(roc_auc_arr)))

        # save the final save_dtfm
        save_dtfm.to_excel('excel_permutations/permutations_{}.xlsx'.format(file_number))

        # plot roc_auc for various for various threasholds
        plt_data = pd.DataFrame({
                "Accuracy":m_acc,
                "Recall": m_rec,
                "Precision": m_pre,
                "ROC AUC": m_auc,
                })

        # print best and worst permutations
        if logger:
            print("Best Permutations:\n{}".format(best_permutations))
            print("Worst Permutations:\n{}".format(worst_permutations))

        # create box plot
        ax = sns.boxplot(data=plt_data)
        ax = sns.swarmplot(data=plt_data, color=".25")

        plt.tight_layout()
        if save_fig:
            plt.savefig('catplot_seven_permutations')
        else:
            plt.show()

