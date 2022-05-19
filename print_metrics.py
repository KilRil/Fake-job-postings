import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, \
    precision_recall_curve, \
    accuracy_score, \
    plot_confusion_matrix, \
    confusion_matrix, \
    ConfusionMatrixDisplay


def calculate_discrete_results(target_val, predicted):
    print("Accuracy:", round(accuracy_score(target_val, predicted), 2))
    print("F1:")
    print("Micro:", round(f1_score(target_val, predicted, average="micro"), 2))
    print("Macro:", round(f1_score(target_val, predicted, average="macro"), 2))
    print("Weighted:", round(f1_score(target_val, predicted, average="weighted"), 2))
    print("For fraudulent:", round(f1_score(target_val, predicted, average="binary"), 2))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(target_val, predicted)).plot(cmap='cividis', ax=ax)
    plt.show()


def calculate_analog_results(target_val, predicted):
    max_score = 0
    max_threshold = None
    max_threshold_predicted_labels = []

    # As labels
    for threshold in np.linspace(0.1, 0.9, 9):
        predicted_labels = []
        for p in predicted:
            if p >= threshold:
                predicted_labels.append([1])
            else:
                predicted_labels.append([0])

        f1_binary = f1_score(target_val, predicted_labels, average="binary")
        if f1_binary > max_score:
            max_threshold = threshold
            max_score = f1_binary
            max_threshold_predicted_labels = predicted_labels

    print("Max threshold", max_threshold)
    calculate_discrete_results(target_val, max_threshold_predicted_labels)
