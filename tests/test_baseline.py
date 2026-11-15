import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from metonymy.baselines import majority_label, predict_majority
from metonymy.metrics import classification_metrics


def test_train_majority_and_constant_predictions():
    fitted = majority_label([1, 1, 0])
    assert fitted == 1
    assert predict_majority(fitted, 4) == [1, 1, 1, 1]


def test_tie_is_literal():
    assert majority_label([1, 0]) == majority_label([0, 1]) == 0


def test_empty_train():
    with pytest.raises(ValueError, match="TRAIN"):
        majority_label([])


def test_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert metrics["accuracy"] == .5
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["LITERAL"]["precision"] == .5
    assert metrics["per_class"]["LITERAL"]["recall"] == 1
    assert metrics["per_class"]["METONYMIC"]["f1"] == 0
    assert metrics["confusion_matrix"] == [[2, 0], [2, 0]]
    assert classification_metrics([0], [0])["macro_f1"] == .5


@pytest.mark.parametrize("truth,pred", [([], []), ([0], []), ([2], [0])])
def test_invalid_metrics(truth, pred):
    with pytest.raises(ValueError):
        classification_metrics(truth, pred)


def test_cli_train_only_and_saved_result(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for split, labels in {"train": [1, 1, 0], "dev": [0, 0, 0]}.items():
        rows = [{"example_id": f"{split}-{i}", "text": f"Paris {i}", "mention": "Paris",
                 "label": label, "language": "en"} for i, label in enumerate(labels)]
        (tmp_path / f"{split}.json").write_text(json.dumps(rows))
    command = [sys.executable, str(root / "scripts/run_majority_baseline.py"),
               "--train", str(tmp_path / "train.json"), "--dev", str(tmp_path / "dev.json"),
               "--output-dir", str(tmp_path / "results")]
    run = subprocess.run(command, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert run.returncode == 0, run.stderr
    result = json.loads((tmp_path / "results/majority_en_dev.json").read_text())
    assert result["majority_label"] == "METONYMIC"
    assert result["evaluation_split"] == "dev"
    assert result["metrics"]["accuracy"] == 0
    assert result["metrics"]["confusion_matrix"] == [[0, 3], [0, 0]]
    command[command.index("--dev") + 1] = str(tmp_path / "train.json")
    run = subprocess.run(command, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert run.returncode == 2
    assert "overlap" in run.stderr
