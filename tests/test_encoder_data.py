"""Synthetic tests for the Group 1 prepared-data bridge; no models needed."""
import pytest

from metonymy.group1.passages import passage
from metonymy.group1.prepare import prepare_hipe_file

HEADER = "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-FINE-LIT\tNE-FINE-METO\tNE-FINE-COMP\tNE-NESTED\tNEL-LIT\tNEL-METO\tMISC\n"


def row(token, lit="O", meto="O", misc="_"):
    return "\t".join([token, lit, meto, "O", "O", "O", "O", "_", "_", misc]) + "\n"


def document(name, rows):
    return (f"# hipe2022:document_id = {name}\n# hipe2022:language = fr\n"
            "# hipe2022:dataset = hipe2020\n# hipe2022:date = 1900-01-01\n" + "".join(rows))


def source(tmp_path, text):
    path = tmp_path / "source.tsv"
    path.write_text(HEADER + text, encoding="utf-8")
    return path


def test_context_boundaries_ids_and_full_mention(tmp_path):
    rows = [row("before") for _ in range(130)]
    rows += [row("New", "B-loc"), row("York", "I-loc")]
    rows += [row("after") for _ in range(140)]
    path = source(tmp_path, document("one", rows) + "\n" + document("two", [row("Paris", "B-loc")]))
    first, second = prepare_hipe_file(path, language="fr", split="train")
    assert first["id"] == "hipe2020-v2.1:fr:one:130-132"
    assert (first["doc_start"], first["doc_end"]) == (130, 132)
    assert (first["start"], first["end"]) == (128, 130)
    assert len(first["tokens"]) == 258  # 128 each side + two target tokens
    assert first["tokens"][128:130] == ["New", "York"]
    assert second["tokens"] == ["Paris"]  # Never crosses a document boundary.
    assert second["date"] == "1900-01-01"
    assert first["group"] == "one"


@pytest.mark.parametrize("meto,label", [("O", 0), ("B-loc", 1), ("B-org", 1)])
def test_labels_partial_and_spacing(tmp_path, meto, label):
    path = source(tmp_path, document("one", [row("Paris", "B-loc", meto, "NoSpaceAfter|Partial-0:4"), row(".")]))
    result, = prepare_hipe_file(path, language="fr", split="dev")
    assert result["label"] == label
    assert result["partial"] is True
    assert result["tokens"][0] == "Paris"  # No Partial slicing.
    rendered = passage(result, markers=None)
    assert rendered["text"] == "Paris."
    assert rendered["text"][rendered["name_start"]:rendered["name_end"]] == "Paris"


def test_unknown_malformed_run_is_not_silently_repaired(tmp_path):
    path = source(tmp_path, document("unknown", [row("Paris", "B-loc"), row("modern", "I-org")]))
    with pytest.raises(ValueError, match="Unreviewed malformed"):
        prepare_hipe_file(path, language="fr", split="train")


def test_reviewed_france_preserved_with_note(tmp_path):
    rows = [row("word") for _ in range(1461)] + [row("France", "B-loc"), row("moderne", "I-org")]
    path = source(tmp_path, document("GDL-1948-04-15-a-i0035", rows))
    result, = prepare_hipe_file(path, language="fr", split="train")
    assert result["doc_end"] == 1462
    assert result["label"] == 0
    assert "malformed IOB2" in result["note"]
    with pytest.raises(ValueError, match="Unreviewed malformed"):
        prepare_hipe_file(path, language="fr", split="dev")


@pytest.mark.parametrize("language,split", [("en", "train"), ("fr", "test")])
def test_rejects_out_of_scope_splits(tmp_path, language, split):
    with pytest.raises(ValueError, match="TRAIN and DEV"):
        prepare_hipe_file(tmp_path / "unused", language=language, split=split)
