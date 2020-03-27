"""
This file will be used to do some
exploratory data analysis on the data

"""

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sea
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from evaluate_model import evaluate_model, performance_assessor
    from confusion_matrix import plot_confusion_matrix
    from sklearn import preprocessing
    from sklearn.metrics import confusion_matrix
    from test_threasholds import find_best_params
    import graphviz
    from tqdm import tqdm

    #run time options
    logger = False
    variable_boxplot = True
    decision_tree = True
    predictors_100 = True
    save_fig=False

    # read in all dataframes and combine
    dtfms = []
    for i in range(11):
        print("Fetching excel_permutations_{}.xlsx".format(i))
        dtfms.append(pd.read_excel('excel_permutations/permutations_{}.xlsx'.format(i), sheet_name='Sheet1', index_col=0))

    # loop and join dataframes
    dtfm = pd.DataFrame()
    for d in dtfms:
        dtfm = dtfm.append(d)

    # clear the variables
    dtfms = None

    # print dtfm head
    if logger:
        print('Head:\n{}'.format(dtfm.head()))
        print("Mean Values:\n{}".format(dtfm.mean()))

    # specify variable list
    var_list = ['AI','PI','ALTO','FRAG_CRO','MOT_PRE','MOT_POS','CONC_CAMARA','VF','AD','VAP','VSL','VCL','ALH','BCF','STR','LIN','MOTILE_PCT','PROGRESSIVE_PCT','RAPID_PCT','MEDIUM_PCT', 'SLOW_PCT','STATIC_PCT']
    # boxplot of separate dtfms
    if variable_boxplot:
        dtfm_plot = pd.DataFrame()
        for v in tqdm(var_list):
            dtfm_v = pd.DataFrame()
            for i in range(7):
                is_v =  dtfm['Var {}'.format(i)] == v
                is_v_dtfm = dtfm[is_v]
                dtfm_v = dtfm_v.append(is_v_dtfm)
            # create a datadrame containing only variable used in loop
            v_array = [v for i in range(len(dtfm_v))]
            dtfm_v['Var'] = v_array

            dtfm_plot = dtfm_plot.append(dtfm_v)

        ax = sea.boxplot(y='AUC', x='Var', data=dtfm_plot)
        plt.xticks(rotation=90)
        if save_fig:
            plt.savefig('permutations_boxplot')
        else:
            plt.show()

    # 100 best predictors
    if predictors_100:
        # sort values
        dtfm_100 = dtfm.nlargest(17000, 'AUC')

        # tot tally
        tot_tally = {}
        for i in tqdm(range(7)):
            val_counts = dtfm_100['Var {}'.format(i)].value_counts()
            # add to tot tally of value counts
            for k in val_counts.keys():
                if tot_tally.get(k) == None:
                    tot_tally[k] = 0

                # increment value by 1
                tot_tally[k] = tot_tally[k] + val_counts[k]

        # prep a dataframe
        dtfm_tot_tally = pd.DataFrame(
                tot_tally.values(),
                index=tot_tally.keys(),
                columns=['Tally'])
        # sort values
        dtfm_tot_tally = dtfm_tot_tally.sort_values(by=['Tally'])
        # plot bar
        dtfm_tot_tally.plot.bar(y='Tally')
        plt.xticks(rotation=90)
        plt.ylabel('Number of occurences in top 10% of permutations')
        if save_fig:
            plt.savefig('permutations_100_countplot')
        else:
            plt.show()


    # decision with target variables on main dataset
    if decision_tree:
        # Set random seed to ensure reproducible runs
        RSEED = 30

        # copy dataframe
        dtfm_tree = dtfm.drop(columns=['Accuracy', 'Precision', 'Recall'])

        # Update Labels in
        dtfm_tree['AUC'] = dtfm_tree['AUC'].where(dtfm_tree['AUC'] >= dtfm['AUC'].mean(), other=0)
        dtfm_tree['AUC'] = dtfm_tree['AUC'].where(dtfm_tree['AUC'] < dtfm_tree['AUC'].mean(), other=1)

        # encode categorical vars
        le = preprocessing.LabelEncoder()
        le.fit(var_list)
        for i in range(7):
            dtfm_tree['Var {}'.format(i)] = le.transform(dtfm_tree['Var {}'.format(i)])

        # Extract the labels
        labels = np.array(dtfm_tree['AUC'])

        #find optimal params
        c_params = find_best_params(dtfm_tree, 'AUC', classifier=DecisionTreeClassifier,
         test_size=0.3, random_state=RSEED, logger=logger)

        # 30% examples in test data
        train, test, train_labels, test_labels = train_test_split(dtfm_tree, labels, stratify = labels, test_size = 0.3, random_state = RSEED)

        # Features for feature importances
        features = list(train.columns)

        if logger:
            print("Train Shape: {}".format(train.shape))
            print("Test Shape: {}".format(test.shape))

        # Make a decision tree and train
        tree = DecisionTreeClassifier(
                max_features=c_params['max_features'],
                max_leaf_nodes=c_params['max_leaf_nodes'],
                min_samples_split=c_params['min_samples_split'],
                random_state=RSEED)

        # Train tree
        tree.fit(train, train_labels)
        if logger:
            print('Decision tree has {} nodes with maximum depth {}.'.format(tree.tree_.node_count, tree.tree_.max_depth))

        # Make probability predictions
        train_probs = tree.predict_proba(train)[:, 1]
        probs = tree.predict_proba(test)[:, 1]

        train_predictions = tree.predict(train)
        predictions = tree.predict(test)

        # evaluate model
        evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels, title='Tree ROC Curve')

        # print other metrics
        performance_assessor(predictions, probs, train_predictions, train_probs, test_labels, train_labels, logger=True)

        # Plot confusion matrix
        cm = confusion_matrix(test_labels, predictions)

        print("Confusion Matrix:\n{}".format(cm))
        # display example decision tree
        export_graphviz(tree, out_file='perm_tree.dot',
                        filled=True, rounded=True,
                        special_characters=True,
                        feature_names=features)

        print('\033[94m' + "To view decision tree example run the following command in terminal:\ndot -Tpng perm_tree.dot -o perm_tree.png" + '\033[0m')

        for i in var_list:
            print("{0} is encoded as {1}".format(i, le.transform([i])))


