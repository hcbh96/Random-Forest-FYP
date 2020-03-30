"""
In this file I will:
    Use a Randomised Search to find optimal RF params
    Train A RF with optimised params
    Evaluate Accuracy
    Evaluate Sensitivity
    Evaluate Precision
    Evaluate ROC AUC
    Determine the features of greatest importance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from confusion_matrix import plot_confusion_matrix
from evaluate_model import evaluate_model, performance_assessor
from sklearn.metrics import confusion_matrix

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

# print other metrics
performance_assessor(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs, test_labels, train_labels, logger=True)

# Plot confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                       title = 'Optimised Forest Confusion Matrix')
