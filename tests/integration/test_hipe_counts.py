"""Opt-in pinned-data assertions, separate from portable synthetic unit tests."""
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("HIPE_REAL_DATA_TESTS") != "1",
                                reason="Set HIPE_REAL_DATA_TESTS=1 to audit local pinned HIPE files")


def test_pinned_counts_and_review_cases():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("audit_hipe", root / "scripts/audit_hipe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit(root / "data/raw/HIPE-2022-data")
    assert report["all_expected_counts_match"]
    assert report["raw_files_match_pinned_revision"]
    assert report["combined"] == {
        "train": {"LITERAL": 4059, "METONYMIC": 770, "total": 4829},
        "dev": {"LITERAL": 1171, "METONYMIC": 191, "total": 1362},
        "test": {"LITERAL": 1220, "METONYMIC": 229, "total": 1449},
    }
    flagged = [m for row in report["splits"] for m in row["flagged_candidates"]]
    assert {(m["document_id"], m["source_token_start"], m["source_token_end"], m["label_name"])
            for m in flagged} == {
        ("GDL-1948-04-15-a-i0035", 1461, 1462, "LITERAL"),
        ("luxwort-1938-05-20-a-i0023", 26, 27, "METONYMIC"),
    }
    assert sum(len(r["review_issues"]) for r in report["splits"]) == 2
    assert all(r["strict_complete"] == (r["split"] != "train") for r in report["splits"])
