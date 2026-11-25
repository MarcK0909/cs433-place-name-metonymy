"""Shared helpers: file paths, readers for the raw files, the prepared data, scores and run logs."""
import hashlib
import json
import math
import os
import platform
import re
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


def attribute(name, text):
    """Value of one XML attribute, or "-" when it is missing."""
    match = re.search(name + r'="([^"]*)"', text)
    return match.group(1) if match else "-"


def read_semeval(path):
    """Return {sample id: fields} of a SemEval XML file, in file order.

    reading, metotype and notes are the original SemEval annotation. label is
    the extra <label> element of SemEval.relabelled.xml ("-" in other files).
    """
    samples = {}
    xml = path.read_text(encoding="iso-8859-1")
    for sample_id, body in re.findall(
        r'<sample id="([^"]+)"[^>]*>(.*?)</sample>', xml, re.S
    ):
        attributes, name = re.search(
            r"<location([^>]*)>(.*?)</location>", body, re.S
        ).groups()
        label = re.search(r"<label>(.*?)</label>", body)
        # Plain text with the place name marked, for showing examples.
        text = re.sub(r"<bnc:title>.*?</bnc:title>", " ", body, flags=re.S)
        text = re.sub(r"<label>.*?</label>", " ", text)
        text = re.sub(r"<location[^>]*>(.*?)</location>", r"[[\1]]", text, flags=re.S)
        text = " ".join(re.sub(r"<[^>]+>", " ", text).split())
        samples[sample_id] = {
            "name": name.strip(),
            "reading": attribute("reading", attributes),
            "metotype": attribute("metotype", attributes),
            "notes": attribute("notes", attributes),
            "label": label.group(1).lower() if label else "-",
            "text": text,
        }
    return samples


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


def save_examples(examples, name):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / f"{name}.jsonl").open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_examples(name, directory=None):
    """Load <directory>/<name>.jsonl (default data/processed), e.g. load_examples("relocar_train")."""
    with (Path(directory or PROCESSED) / f"{name}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def target_name(example):
    return " ".join(example["tokens"][example["start"] : example["end"]])


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


def bootstrap_interval(labels, predictions, groups, metric="macro_f1", n_samples=1000, seed=0):
    """95% interval of a score, resampling whole groups (sentences or articles).

    Mentions from the same group are not independent, so they are drawn together.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    _, group_index = np.unique(np.asarray(groups), return_inverse=True)
    members = [[] for _ in range(group_index.max() + 1)]
    for row, group in enumerate(group_index):
        members[group].append(row)
    members = [np.array(rows) for rows in members]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_samples):
        chosen = rng.integers(len(members), size=len(members))
        rows = np.concatenate([members[group] for group in chosen])
        values.append(scores(labels[rows], predictions[rows])[metric])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def mcnemar_exact(labels, predictions_a, predictions_b):
    """Exact two-sided McNemar test of two systems on the same mentions.

    Returns (a_only, b_only, p): mentions only system A gets right, mentions only
    system B gets right, and the p-value of the binomial test on these two counts.
    """
    labels = np.asarray(labels)
    correct_a = np.asarray(predictions_a) == labels
    correct_b = np.asarray(predictions_b) == labels
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n
    return a_only, b_only, min(1.0, 2 * tail)


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
