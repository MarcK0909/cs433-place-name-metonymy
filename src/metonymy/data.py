"""Prepared mention-level data loading; no inferred splits or HIPE semantics."""
from __future__ import annotations

import csv
import json
import random
import statistics
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

LABEL_NAMES = {0: "LITERAL", 1: "METONYMIC"}


def normalize_label(value: Any) -> int:
    """Accept only explicit binary integers or their documented names."""
    if type(value) is int and value in LABEL_NAMES:
        return value
    if isinstance(value, str) and value in ("0", "1", "LITERAL", "METONYMIC"):
        return {"0": 0, "1": 1, "LITERAL": 0, "METONYMIC": 1}[value]
    raise ValueError(f"Invalid/unexpected label {value!r}; expected 0/LITERAL or 1/METONYMIC")


def normalize_record(record: Any) -> dict[str, Any]:
    """Validate required fields, preserving text and additional metadata verbatim."""
    if not isinstance(record, dict):
        raise ValueError("Malformed record: expected an object")
    result = dict(record)
    for field in ("example_id", "text", "mention", "language"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Missing or empty {field}: expected a nonempty string")
    label = normalize_label(record.get("label", record.get("label_name")))
    if "label_name" in record and record["label_name"] != LABEL_NAMES[label]:
        raise ValueError("label_name conflicts with label or is not LITERAL/METONYMIC")
    if record.get("document_id") is not None and not isinstance(record["document_id"], str):
        raise ValueError("document_id must be a string or null")
    result.update(label=label, label_name=LABEL_NAMES[label], document_id=record.get("document_id"))
    return result


def read_records(path: str | Path) -> list[Any]:
    """Read JSON arrays, JSONL, CSV or TSV without changing source files."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".json":
            records = json.load(stream)
            if not isinstance(records, list):
                raise ValueError(f"{path}: expected a JSON array")
            return records
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in stream if line.strip()]
        if path.suffix.lower() in (".csv", ".tsv"):
            reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            if "TOKEN" in (reader.fieldnames or []):
                raise ValueError("Raw HIPE token data are unsupported: inspect actual annotations before implementing extraction")
            return list(reader)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def duplicate_issues(records: list[dict[str, Any]]) -> list[str]:
    """Report repeated IDs and duplicate content even when IDs differ."""
    issues = []
    ids = Counter(r["example_id"] for r in records)
    for example_id, count in ids.items():
        if count > 1:
            issues.append(f"Duplicate example ID {example_id!r} ({count} records)")
    keys = Counter((r["text"], r["mention"], r["label"], r["language"], r["document_id"]) for r in records)
    duplicates = sum(count - 1 for count in keys.values())
    if duplicates:
        issues.append(f"Exact duplicate examples (excluding IDs/additional metadata): {duplicates}")
    return issues


def load_dataset(path: str | Path, language: str = "en") -> list[dict[str, Any]]:
    """Strictly validate all input rows, then select one explicitly requested language."""
    records = []
    for index, raw in enumerate(read_records(path), 1):
        try:
            records.append(normalize_record(raw))
        except ValueError as error:
            raise ValueError(f"{path}: record {index}: {error}") from error
    for issue in duplicate_issues(records):
        warnings.warn(issue, UserWarning, stacklevel=2)
    return [r for r in records if r["language"] == language]


def overlap_issues(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Identify example-ID leakage across supplied splits."""
    issues = []
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = {r["example_id"] for r in splits[left]} & {r["example_id"] for r in splits[right]}
            if overlap:
                issues.append(f"Example-ID overlap {left}/{right}: {len(overlap)}; IDs: {sorted(overlap)[:10]}")
    return issues


def inspect_dataset(path: str | Path, language: str = "en", seed: int = 433) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Report malformed rows explicitly; statistics cover valid selected-language rows."""
    raw_records = read_records(path)
    valid, issues = [], []
    for index, raw in enumerate(raw_records, 1):
        try:
            valid.append(normalize_record(raw))
        except ValueError as error:
            issues.append(f"Record {index}: {error}")
    issues.extend(duplicate_issues(valid))
    selected = [r for r in valid if r["language"] == language]
    counts = Counter(r["label_name"] for r in selected)
    if selected and max(counts.values()) / len(selected) >= .9:
        issues.append("Severe class imbalance: at least 90% of examples belong to one class")
    if not selected:
        issues.append(f"No valid examples for language {language!r}")
    def lengths(field: str) -> dict[str, float | int | None]:
        values = [len(r[field]) for r in selected]
        return {"min": min(values) if values else None, "max": max(values) if values else None,
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None}
    rng = random.Random(seed)
    report = {
        "path": str(path), "language": language, "seed": seed,
        "input_records": len(raw_records), "invalid_records": len(raw_records) - len(valid),
        "valid_records_other_languages": len(valid) - len(selected),
        "available_languages": dict(Counter(r["language"] for r in valid)),
        "total_examples": len(selected),
        "classes": {name: {"count": counts[name], "percentage": 100 * counts[name] / len(selected) if selected else 0.0}
                    for name in LABEL_NAMES.values()},
        "unique_mentions": len({r["mention"] for r in selected}),
        "text_length_characters": lengths("text"), "mention_length_characters": lengths("mention"),
        "samples": {name: rng.sample([r for r in selected if r["label_name"] == name], min(5, counts[name]))
                    for name in LABEL_NAMES.values()}, "warnings": issues,
    }
    return report, selected
