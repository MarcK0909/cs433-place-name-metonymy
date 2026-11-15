"""Fit on TRAIN and evaluate on DEV by default; TEST requires explicit opt-in."""
import argparse
import json
import re
from pathlib import Path

from metonymy.baselines import majority_label, predict_majority
from metonymy.data import LABEL_NAMES, load_dataset, overlap_issues
from metonymy.metrics import classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Official/project TRAIN file")
    parser.add_argument("--dev", type=Path, help="Official/project DEV file")
    parser.add_argument("--test", type=Path, help="Official/project TEST file (only for final evaluation)")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines"))
    args = parser.parse_args()
    evaluation_path = getattr(args, args.split)
    if evaluation_path is None:
        parser.error(f"--{args.split} is required for {args.split.upper()} evaluation")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.language):
        parser.error("--language must be a language code, e.g. en")
    try:
        train = load_dataset(args.train, args.language)
        evaluation = load_dataset(evaluation_path, args.language)
        issues = overlap_issues({"train": train, args.split: evaluation})
        if issues:
            raise ValueError("; ".join(issues) + "; resolve split provenance before evaluation")
        label = majority_label([r["label"] for r in train])
        metrics = classification_metrics([r["label"] for r in evaluation], predict_majority(label, len(evaluation)))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Baseline cannot run: {error}\n")
    result = {"baseline": "majority", "language": args.language, "train_size": len(train),
              "evaluation_size": len(evaluation), "evaluation_split": args.split,
              "majority_label": LABEL_NAMES[label], "majority_label_id": label,
              "tie_rule": "LITERAL (0)", "train_path": str(args.train),
              "evaluation_path": str(evaluation_path), "metrics": metrics}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"majority_{args.language}_{args.split}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Macro-F1: {metrics['macro_f1']:.6f}")
    print(json.dumps(result, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
