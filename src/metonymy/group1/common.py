"""Shared helpers: file paths, readers for the raw files, the prepared data, scores and run logs."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Hugging Face tokenizers warn after a fork (e.g. the git call in run_info).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs"


# ---------- Readers for the raw files ----------


def read_hipe(path):
    """Return the documents of a HIPE TSV file: id, date and token rows of each.

    Each row is the list of the 10 columns: TOKEN, NE-COARSE-LIT,
    NE-COARSE-METO, ..., MISC.
    """
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:  # comment or empty line
            if line.startswith("# hipe2022:document_id"):
                documents.append({"id": line.split("=")[1].strip(), "date": "", "rows": []})
            elif line.startswith("# hipe2022:date"):
                documents[-1]["date"] = line.split("=")[1].strip()
        elif not line.startswith("TOKEN\t"):
            documents[-1]["rows"].append(line.split("\t"))
    return documents


def spans(tags):
    """Turn IOB tags (B-loc, I-loc, O) into (start, end, type) spans."""
    result = []
    start = None
    for position, tag in enumerate(tags + ["O"]):
        if start is not None and not tag.startswith("I-"):
            result.append((start, position, tags[start][2:]))
            start = None
        if tag.startswith("B-"):
            start = position
    return result


# ---------- Prepared data (one JSON object per line) ----------


def load_examples(name, directory=None):
    """Load <directory>/<name>.jsonl (default data/processed), e.g. load_examples("relocar_train")."""
    with (Path(directory or PROCESSED) / f"{name}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


# ---------- Scores ----------


def scores(labels, predictions):
    """Accuracy, macro-F1 and precision/recall/F1 of both classes (0 literal, 1 metonymic)."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    result = {"accuracy": float(np.mean(labels == predictions))}
    f1_values = []
    for value, name in [(0, "literal"), (1, "metonymic")]:
        true_positives = np.sum((predictions == value) & (labels == value))
        predicted = np.sum(predictions == value)
        actual = np.sum(labels == value)
        precision = true_positives / predicted if predicted else 0.0
        recall = true_positives / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"precision_{name}"] = float(precision)
        result[f"recall_{name}"] = float(recall)
        result[f"f1_{name}"] = float(f1)
        f1_values.append(f1)
    result["macro_f1"] = float(np.mean(f1_values))
    return result


# ---------- Run logs ----------


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def run_info():
    """What a run needs for the report: versions, machine, time and code version."""
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.node(),
        "numpy": np.__version__,
    }
    for module in ["torch", "transformers"]:
        if module in sys.modules:
            info[module] = sys.modules[module].__version__
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = "not a git repository"
    return info


def data_versions(names, directory=None):
    """md5 of each data file a run used, to check later that the data did not change."""
    return {name: file_md5(Path(directory or PROCESSED) / f"{name}.jsonl") for name in names}
