import json

import pytest

from metonymy.data import inspect_dataset, load_dataset, normalize_record, overlap_issues


def record(**changes):
    return dict({"example_id": "a", "text": "In [TGT] Paris [/TGT].", "mention": "Paris",
                 "label": "LITERAL", "language": "en"}, **changes)


def write_records(tmp_path, records):
    path = tmp_path / "data.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


@pytest.mark.parametrize("label", [2, -1, True, 1.0, "unknown", None])
def test_invalid_labels(label):
    with pytest.raises(ValueError, match="label"):
        normalize_record(record(label=label))


@pytest.mark.parametrize("field", ["text", "mention", "example_id", "language"])
def test_malformed_records(field):
    with pytest.raises(ValueError, match=field):
        normalize_record(record(**{field: ""}))


def test_normalization_and_language_filter(tmp_path):
    path = write_records(tmp_path, [record(), record(example_id="b", language="fr")])
    rows = load_dataset(path)
    assert len(rows) == 1
    assert rows[0]["label"] == 0
    assert rows[0]["document_id"] is None
    assert rows[0]["text"] == record()["text"]


def test_duplicates(tmp_path):
    path = write_records(tmp_path, [record(), record()])
    with pytest.warns(UserWarning, match="Duplicate example ID"):
        rows = load_dataset(path)
    assert len(rows) == 2
    report, _ = inspect_dataset(path)
    assert any("Exact duplicate" in issue for issue in report["warnings"])


def test_inspection_reports_invalid_rows(tmp_path):
    path = write_records(tmp_path, [record(), record(label="wrong"), record(text="")])
    report, _ = inspect_dataset(path)
    assert report["input_records"] == 3
    assert report["invalid_records"] == 2
    assert report["total_examples"] == 1
    assert report["classes"]["LITERAL"]["percentage"] == 100
    assert report == inspect_dataset(path)[0]
    with pytest.raises(ValueError, match="record 2"):
        load_dataset(path)


def test_overlap():
    assert overlap_issues({"train": [record()], "dev": [record()]})


@pytest.mark.parametrize("suffix,contents", [
    (".jsonl", json.dumps(record()) + "\n"),
    (".csv", "example_id,text,mention,label,language\na,Paris.,Paris,0,en\n"),
    (".tsv", "example_id\ttext\tmention\tlabel\tlanguage\na\tParis.\tParis\t0\ten\n"),
])
def test_formats(tmp_path, suffix, contents):
    path = tmp_path / ("data" + suffix)
    path.write_text(contents)
    assert load_dataset(path)[0]["label_name"] == "LITERAL"


def test_raw_hipe_rejected(tmp_path):
    path = tmp_path / "raw.tsv"
    path.write_text("TOKEN\tNE-COARSE-LIT\nParis\tB-loc\n")
    with pytest.raises(ValueError, match="Raw HIPE"):
        load_dataset(path)
