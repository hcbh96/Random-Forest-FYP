import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from evaluate_model import evaluate_model, performance_assessor
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, multilabel_confusion_matrix
from collections import Counter
from confusion_matrix import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Set random seed to ensure reproducible runs
RSEED = 30

dtfm=pd.read_excel('cleaned_data.xlsx', sheet_name='Sheet1')

#Remove columns not to be used in modelling
dtfm = dtfm.drop(columns=['ORDEM','DATA','AMOSTRA','REPLICATA','ANIMAL','PARTIDA','CLIV','CELLS_COUNT'])


print("Describe Output Vars: \n {}".format(dtfm["BLAST_D8"].describe()))
"""
One of the thigns i need to do is categorise the output data

Where:
- 0 is bad quality 0 - 50%
- 1 is good quality 50 - 100%

I will use the following statistics to make the decsion:

Statistics for each column after outlier removal
             CLIV    BLAST_D8  CELLS_COUNT
count  313.000000  313.000000   180.000000
mean    72.070374   21.475320   171.115891
std      8.942164   11.093061    42.876076
min     49.350649    0.000000    57.000000
25%     65.079365   12.121212   144.875000
50%     72.151899   20.312500   169.875000
75%     79.487179   29.629630   195.437500
max     90.140845   53.623188   269.000000


For BLAST_D8:
    0 < 21.475320
    1 >= 21.475320

"""
# Update Labels in Blast_D8 and CLIV

dtfm['BLAST_D8'] = dtfm['BLAST_D8'].where(dtfm['BLAST_D8'] >= 21.475320, other=0)
dtfm['BLAST_D8'] = dtfm['BLAST_D8'].where(dtfm['BLAST_D8'] < 21.475320, other=1)


# Make a copy for dtfm blast
print("Blast_D8 value counts:\n {}".format(dtfm['BLAST_D8'].value_counts()))


# Extract the labels
labels = np.array(dtfm.pop('BLAST_D8'))

# 30% examples in test data
train, test, train_labels, test_labels = train_test_split(dtfm, labels, stratify = labels, test_size = 0.3, random_state = RSEED)

#imputation of missing values
train = train.fillna(train.mean())
test = test.fillna(test.mean())

# Features for feature importances
features = list(train.columns)

print("Train Shape: {}".format(train.shape))
print("Test Shape: {}".format(test.shape))


"""
Train decision tree on data with unlimited depth to check for overfitting
"""

# Make a decision tree and train
tree = DecisionTreeClassifier(random_state=RSEED)

# Train tree
tree.fit(train, train_labels)
print('Decision tree has {} nodes with maximum depth {}.'.format(tree.tree_.node_count, tree.tree_.max_depth))


"""
Assess decision tree performance

I would expect this to overfit but we want to make sure
"""

# Make probability predictions
train_probs = tree.predict_proba(train)[:, 1]
probs = tree.predict_proba(test)[:, 1]

train_predictions = tree.predict(train)
predictions = tree.predict(test)

# evaluate model
evaluate_model(predictions, probs, train_predictions, train_probs, test_labels, train_labels, title='Tree ROC Curve')

# Plot confusion matrix
cm = confusion_matrix(test_labels, predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                      title = 'Tree Confusion Matrix')
"""
Confusion Matrix:
[[35  9]
 [10 35]]
Classification Accuracy: 0.7865168539325843
Classification Sensitivity: 0.7865168539325843

From a single with a confusion matrix we can see above Accuracy and Sesitivity

These should form our base projection or possibly projections from Mayra?

Should we instead maybe take two classes as this would allow the plotting of
ROC curves etc -

Mayra mentioned that

**The idea with this project more than predict embryo production is to see if there is any variables from sperm analysis that can predict these production.
That's why we used so many bulls. Ore research is based on these ideas, the bull effect, which sperm analysis can we do to predict embryo production. **

Consider this when deciding whether to use binary or non binary classification

Let check out feature importance in the decision tree
"""
fi = pd.DataFrame({'feature': features,
                   'importance': tree.feature_importances_}).\
                    sort_values('importance', ascending = False)

print("Features of most importance in decision tree: \n{}".format(fi.head()))

"""
This porucdes the following results

Features of most importance in decision tree:
     feature  importance
17       ALH    0.151271
3   SUB_3_LS    0.145387
8   FRAG_CRO    0.079971
18       BCF    0.077984
20       LIN    0.065810


I want to at some point check co-linearity between the above variables.

"""


"""
Random Forest to check if classification improves
"""
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               random_state=RSEED,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)


# Fit on training data
model.fit(train, train_labels)

#Calculate the avg number of nodes per tree
n_nodes = []
max_depths = []

for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print('Average number of nodes: {}'.format(int(np.mean(n_nodes))))
print('Average maximum depth: {}'.format(int(np.mean(max_depths))))

"""
Average number of nodes 59
Average maximum depth 10

Test results against Base
"""
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

# evaluate model
evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs, test_labels, train_labels, title='Forest ROC')


"""
Produces the following:

Confusion Matrix:
[[41  3]
 [ 7 38]]
Classification Accuracy: 0.8876404494382022
Classification Sensitivity: 0.8876404494382022
"""

fi_model = pd.DataFrame({'feature': features,
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)

print("Features of most importance in RF:\n{}".format(fi_model.head(10)))


"""
Features of most importance in RF:
        feature  importance
5            AI    0.068131
11  CONC_CAMARA    0.060125
12           VF    0.047324
8      FRAG_CRO    0.043618
13           AD    0.043218
26   STATIC_PCT    0.042707
7          ALTO    0.042087
14          VAP    0.040594
1      SUB_1_RP    0.038761
2       SUB_2_H    0.037506

Maybe do a co-linearity check between variables again
"""

"""
Optimising input params
"""

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

n_nodes = []
max_depths = []

for ind_tree in best_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print('Average number of nodes: {}'.format(int(np.mean(n_nodes))))
print('Average maximum depth: {}'.format(int(np.mean(max_depths))))

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs,test_labels, train_labels, title='Optimised Forest ROC Curve')

# Plot confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                       title = 'Optimised Forest Confusion Matrix')
