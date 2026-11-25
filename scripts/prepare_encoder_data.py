"""Prepare the four pinned FR/DE TRAIN/DEV files and verify Group 1 parity."""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metonymy.group1.prepare import prepare_hipe_file


def prepare(data_root: Path, output_dir: Path, *, overwrite: bool = False) -> dict:
    """Validate every source and prepared checksum before writing any dataset."""
    reference = json.loads(files("metonymy.group1").joinpath("data_reference.json").read_text())
    revision = subprocess.check_output(["git", "-C", str(data_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != reference["hipe_revision"]:
        raise ValueError(f"Unexpected HIPE revision: {revision}")
    destinations = [output_dir / f"{name}.jsonl" for name in reference["prepared_md5"]]
    destinations.append(output_dir / "preparation_report.json")
    if not overwrite and any(path.exists() for path in destinations):
        raise ValueError("Output already exists; inspect it or explicitly pass --overwrite")
    prepared_files, records = {}, []
    all_ids = set()
    for name, expected_md5 in reference["prepared_md5"].items():
        _, language, split = name.split("_")
        relative = Path("data/v2.1/hipe2020") / language / f"HIPE-2022-v2.1-hipe2020-{split}-{language}.tsv"
        path = data_root / relative
        raw = path.read_bytes()
        pinned = subprocess.check_output(["git", "-C", str(data_root), "show",
                                           f"{revision}:{relative.as_posix()}"])
        if raw != pinned:
            raise ValueError(f"Raw source differs from pinned revision: {path}")
        examples = prepare_hipe_file(path, language=language, split=split)
        ids = [row["id"] for row in examples]
        if len(set(ids)) != len(ids) or all_ids.intersection(ids):
            raise ValueError(f"Duplicate/overlapping example IDs in {name}")
        all_ids.update(ids)
        counts = {"literal": sum(row["label"] == 0 for row in examples),
                  "metonymic": sum(row["label"] == 1 for row in examples)}
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples).encode("utf-8")
        digest = hashlib.md5(content).hexdigest()
        if counts != reference["counts"][name] or digest != expected_md5:
            raise ValueError(f"{name}: prepared data parity failed; counts={counts}, md5={digest}, expected={expected_md5}")
        prepared_files[name] = content
        records.append({"name": name, "source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "counts": counts, "total": len(examples), "prepared_md5": digest,
                        "matches_group1": True, "partial_retained": sum(row["partial"] for row in examples),
                        "reviewed_targets": [{"id": row["id"], "label": row["label"], "note": row["note"]}
                                             for row in examples if row["note"]]})
    report = {"source_revision": revision, "group1_commit": reference["upstream_commit"],
              "reference_config": reference["reference_config"], "files": records,
              "all_prepared_checksums_match": True, "test_processed": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in prepared_files.items():
        (output_dir / f"{name}.jsonl").write_bytes(content)
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/HIPE-2022-data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/group1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = prepare(args.data_root, args.output_dir, overwrite=args.overwrite)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Preparation stopped: {error}\n")
    for entry in report["files"]:
        print(f"{entry['name']}: {entry['counts']}, total={entry['total']}, Group 1 checksum MATCH")
    print(f"Report: {args.output_dir / 'preparation_report.json'}")


if __name__ == "__main__":
    main()
