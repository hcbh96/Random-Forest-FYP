from sklearn.metrics import recall_score, precision_score, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(predictions, probs, train_predictions, train_probs,
        test_labels, train_labels, title='ROC Plot',
        plot=False, save_fig=False, logger=True):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}

    baseline['recall'] = recall_score(test_labels, [np.random.randint(2) for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, [np.random.randint(2) for _ in range(len(test_labels))])
    baseline['roc'] = 0.5

    results = {}

    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['accuracy'] = accuracy_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)

    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['accuracy'] = accuracy_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)

    if logger:
        for metric in ['recall', 'precision', 'roc']:
            print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    if plot | save_fig:
        plt.figure(figsize = (8, 6))
        plt.rcParams['font.size'] = 16

        # Plot both curves
        plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
        plt.plot(model_fpr, model_tpr, 'r', label = 'model')
        plt.legend();
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(title);
        if save_fig == True:
            plt.savefig(title)
        else:
            plt.show()

    return [baseline, results, train_results]

def performance_assessor(predictions, probs, train_predictions, train_probs, test_labels, train_labels, average='micro', logger=False):
    """Returns metrics assessing the performance of a classifier

    USAGE: performance_assessor(labels, predictions, average='micro', logger=False)

    INPUTS:
      lablels: true labels
      predictions: predicted labels
      (optional)
      average=micro: sensitivity detail
      logger=False: log out put to console

    OUTPUT:
        [
        confusion,
        accuracy,
        sensitivity,
        test_roc_auc,
        baseline_roc_auc,
        ]
    """
    # calc metrics
    test_accuracy = accuracy_score(test_labels, predictions)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_sensitivity = recall_score(test_labels, predictions, average='micro')
    train_sensitivity = recall_score(train_labels, train_predictions, average='micro')
    test_roc_auc = roc_auc_score(test_labels, probs)
    train_roc_auc = roc_auc_score(train_labels, train_probs)

    #log results
    if logger:
        print("Train Accuracy: {}, Test Accuracy: {}".format(train_accuracy, test_accuracy))
        print("Train Sensitivity: {}, Test Sensitivity: {}".format(train_sensitivity, test_sensitivity))
        print('Train ROC AUC  Score: {}, Test ROC AUC Score: {}'.format(train_roc_auc, test_roc_auc))


    return [
            train_accuracy,
            test_accuracy,
            train_sensitivity,
            test_sensitivity,
            train_roc_auc,
            test_roc_auc,
            ]

