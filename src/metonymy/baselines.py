"""A majority baseline fitted exclusively on supplied TRAIN labels."""
from collections import Counter
from collections.abc import Iterable

from metonymy.data import normalize_label


def majority_label(train_labels: Iterable[int]) -> int:
    """Return the most frequent TRAIN label; ties select LITERAL (0)."""
    counts = Counter(normalize_label(label) for label in train_labels)
    if not counts:
        raise ValueError("TRAIN has no examples; cannot determine majority class")
    return max((0, 1), key=lambda label: counts[label])


def predict_majority(label: int, size: int) -> list[int]:
    """Predict the fitted label for each evaluation example."""
    label = normalize_label(label)
    if size < 0:
        raise ValueError("Prediction size must be nonnegative")
    return [label] * size
