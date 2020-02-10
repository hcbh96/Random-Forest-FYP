"""
In this file I want to:
    create DT
    Train DT
    Test DT
    Analyse Accurancy
    Analyse Sensitivity
    Analyse Precision
    Check Feature Importance
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

