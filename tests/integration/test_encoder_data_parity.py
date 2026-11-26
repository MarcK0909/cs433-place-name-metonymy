"""Opt-in parity against Group 1's recorded CE-run prepared-file checksums."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("HIPE_REAL_DATA_TESTS") != "1",
                                reason="Set HIPE_REAL_DATA_TESTS=1 for pinned real-data parity")


def test_prepared_files_match_recorded_baseline(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("prepare_encoder_data", root / "scripts/prepare_encoder_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data_root = root / "data/raw/HIPE-2022-data"
    output = tmp_path / "prepared"
    report = module.prepare(data_root, output)
    assert report["all_prepared_checksums_match"]
    assert sum(r["total"] for r in report["files"]) == 6191
    assert sum(len(r["reviewed_targets"]) for r in report["files"]) == 2
    assert not report["test_processed"]
    assert not list(output.glob("*test*"))
    assert json.loads((output / "preparation_report.json").read_text()) == report
    with pytest.raises(ValueError, match="Output already exists"):
        module.prepare(data_root, output)
    # A content mismatch must fail before any prepared file is written.
    monkeypatch.setattr(module, "prepare_hipe_file", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="parity failed"):
        module.prepare(data_root, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
