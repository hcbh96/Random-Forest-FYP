"""
In this file I will:
    Build an unoptimised RF
    Evaluate Accuracy of RF
    Evaluate Sensitivity of RF
    Evaluate Precision of RF
    Evaluate ROC AUC of RF
    Evaluate features of greatest importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evaluate_model import evaluate_model
from confusion_matrix import plot_confusion_matrix
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


# Plot confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                       title = 'Unoptimised Forest Confusion Matrix')

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
