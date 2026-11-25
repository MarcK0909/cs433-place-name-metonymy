"""Audit only the six pinned, unmasked French/German hipe2020 files."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# Allow the documented `python scripts/audit_hipe.py` without installing package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metonymy.hipe import SOURCE_REVISION, HipeParsingError, extract_mentions, parse_documents

EXPECTED = {
    "fr": {"train": (2644, 445), "dev": (670, 104), "test": (743, 111)},
    "de": {"train": (1415, 325), "dev": (501, 87), "test": (477, 118)},
}


def counts(labels: Counter) -> dict[str, int]:
    return {"LITERAL": labels["LITERAL"], "METONYMIC": labels["METONYMIC"],
            "total": labels["LITERAL"] + labels["METONYMIC"]}


def audit(root: Path) -> dict:
    """Return strict and review-candidate counts, issues and source fingerprints.

    Counts matching the workbench do not resolve annotation issues. No source
    labels are changed; review-mode candidates are never saved as model inputs.
    """
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"Expected revision {SOURCE_REVISION}, found {revision}")
    reports = []
    combined = {split: Counter() for split in ("train", "dev", "test")}
    for language, splits in EXPECTED.items():
        for split, expected in splits.items():
            path = root / "data/v2.1/hipe2020" / language / f"HIPE-2022-v2.1-hipe2020-{split}-{language}.tsv"
            # Verify working-tree bytes too: HEAD alone cannot detect raw edits.
            source = path.read_bytes()
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{SOURCE_REVISION}:{path.relative_to(root).as_posix()}"])
            if source != committed:
                raise ValueError(f"Raw file differs from pinned revision: {path}")
            strict, candidates = Counter(), Counter()
            failures, issues, flagged = [], [], []
            partial_count = 0
            with path.open(encoding="utf-8-sig") as stream:
                for doc in parse_documents(stream, source_path=str(path)):
                    try:
                        mentions = extract_mentions(doc, language=language, split=split)
                        strict.update(r["label_name"] for r in mentions)
                    except HipeParsingError as error:
                        failures.append(str(error))
                        mentions = extract_mentions(doc, language=language, split=split, review_issues=issues)
                    candidates.update(r["label_name"] for r in mentions)
                    partial_count += sum(bool(r["partial"]) for r in mentions)
                    flagged.extend(r for r in mentions if r.get("annotation_issues"))
            observed = counts(candidates)
            expected_counts = dict(zip(("LITERAL", "METONYMIC"), expected))
            expected_counts["total"] = sum(expected)
            reports.append({"language": language, "split": split, "path": str(path),
                            "sha256": hashlib.sha256(source).hexdigest(),
                            "strict_complete": not failures, "strict_successful_documents_only": counts(strict),
                            "strict_failures": failures, "observed_candidates": observed,
                            "expected": expected_counts, "matches": observed == expected_counts,
                            "partial_candidates_retained": partial_count,
                            "review_issues": issues, "flagged_candidates": flagged})
            combined[split].update(candidates)
    return {"source_revision": revision, "raw_files_match_pinned_revision": True,
            "policy": "strict by default; explicit review candidates retain B-type spans before malformed I runs; no tag repair",
            "splits": reports, "combined": {split: counts(labels) for split, labels in combined.items()},
            "all_expected_counts_match": all(r["matches"] for r in reports),
            "requires_annotation_review": any(r["review_issues"] for r in reports)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/HIPE-2022-data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audits/hipe_counts.json"))
    args = parser.parse_args()
    try:
        report = audit(args.data_root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Audit failed: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Language Split LITERAL METONYMIC Total Match Strict-complete")
    for row in report["splits"]:
        c = row["observed_candidates"]
        print(row["language"], row["split"], c["LITERAL"], c["METONYMIC"], c["total"], row["matches"], row["strict_complete"])
        for issue in row["review_issues"]:
            print("  REVIEW", issue["document_id"], issue["column"], issue["source_token_start"], issue["source_token_end"], issue["tag"])
    print("Combined:", report["combined"])
    print(f"Detailed report: {args.output}")
    if not report["all_expected_counts_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
