"""Evaluation with a fixed two-class label order, including absent classes."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from metonymy.data import LABEL_NAMES, normalize_label


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Calculate Macro-F1 and per-class scores; undefined scores are zero."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Evaluation requires nonempty, equally sized truth and prediction lists")
    truth = [normalize_label(value) for value in y_true]
    predictions = [normalize_label(value) for value in y_pred]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f1.mean()), "accuracy": float(accuracy_score(truth, predictions)),
            "per_class": {name: {"precision": float(precision[label]), "recall": float(recall[label]),
                                 "f1": float(f1[label]), "support": int(support[label])}
                          for label, name in LABEL_NAMES.items()},
            "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
            "confusion_matrix_labels": list(LABEL_NAMES.values()),
            "confusion_matrix_orientation": "rows=true, columns=predicted"}
