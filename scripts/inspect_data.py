"""Inspect explicit prepared splits, or inventory the data directory when absent."""
import argparse
import json
from pathlib import Path

from metonymy.data import inspect_dataset, overlap_issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    for split in ("train", "dev", "test"):
        parser.add_argument(f"--{split}", type=Path, help=f"Supervisor-provided {split.upper()} file")
    parser.add_argument("--language", default="en")
    parser.add_argument("--seed", type=int, default=433)
    args = parser.parse_args()
    reports, splits = {}, {}
    try:
        for split in ("train", "dev", "test"):
            path = getattr(args, split)
            if path is not None:
                reports[split], splits[split] = inspect_dataset(path, args.language, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Inspection failed: {error}\n")
    inventory = sorted(str(p) for p in args.data_dir.rglob("*") if p.is_file() and p.name != ".gitkeep")
    issues = overlap_issues(splits)
    if not reports:
        issues.append("No splits supplied. No dataset files found." if not inventory else
                      "Files found; supply explicit --train/--dev/--test paths. Split roles are never inferred.")
    print(json.dumps({"language": args.language, "data_inventory": inventory,
                      "splits": reports, "warnings": issues}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
