"""Synthetic HIPE fixtures; no dependency on downloaded datasets."""
from io import StringIO

import pytest

from metonymy.hipe import HipeParsingError, extract_mentions, load_hipe_mentions, parse_documents

HEADER = "TOKEN\tNE-COARSE-LIT\tNE-COARSE-METO\tNE-NESTED\tMISC\n"
META = "# hipe2022:document_id = doc-1\n# hipe2022:language = fr\n# hipe2022:dataset = hipe2020\n"


def document(rows):
    text = HEADER + META + "\n".join("\t".join(row) for row in rows)
    return next(parse_documents(StringIO(text), source_path="synthetic.tsv"))


def extract(rows):
    return extract_mentions(document(rows), language="fr", split="train")


@pytest.mark.parametrize("kind", [None, "org", "loc", "pers", "prod"])
def test_aligned_labels(kind):
    rows = [("New", "B-loc", f"B-{kind}" if kind else "O", "O", "_"),
            ("York", "I-loc", f"I-{kind}" if kind else "O", "O", "_")]
    result, = extract(rows)
    assert result["mention"] == "New York"
    assert result["label"] == int(kind is not None)
    assert result["label_name"] == ("METONYMIC" if kind else "LITERAL")
    assert result["contextual_type"] == kind
    assert (result["source_token_start"], result["source_token_end"]) == (0, 2)


def test_spacing_and_provenance(tmp_path):
    rows = [("In", "O", "O", "O", "_"),
            ("La", "B-loc", "O", "O", "EndOfLine"),
            ("Chaux", "I-loc", "O", "O", "NoSpaceAfter"),
            ("-", "I-loc", "O", "O", "NoSpaceAfter"),
            ("de", "I-loc", "O", "O", "NoSpaceAfter"),
            ("-", "I-loc", "O", "O", "NoSpaceAfter"),
            ("Fonds", "I-loc", "O", "O", "EndOfSentence")]
    result, = extract(rows)
    assert result["mention"] == "La Chaux-de-Fonds"
    assert result["document_id"] == "doc-1"
    assert result["language"] == "fr"
    assert result["split"] == "train"
    assert result["dataset"] == "hipe2020"
    assert result["dataset_version"] == "v2.1"
    assert result["source_token_start"] == 1
    assert result["source_token_end"] == 7
    assert result["source_line_numbers"] == list(range(6, 12))
    assert result["example_id"] == extract(rows)[0]["example_id"]
    assert "text" not in result
    path = tmp_path / "fixture.tsv"
    path.write_text(HEADER + META + "\n".join("\t".join(r) for r in rows))
    assert load_hipe_mentions(path, language="fr", split="train")[0]["example_id"] == result["example_id"]


@pytest.mark.parametrize("column", [1, 2])
@pytest.mark.parametrize("tags", [("I-loc",), ("B-org", "I-loc"), ("O", "I-loc"), ("_",), ("loc",)])
def test_malformed_bio(column, tags):
    rows = []
    for tag in tags:
        row = ["word", "O", "O", "O", "_"]
        row[column] = tag
        rows.append(row)
    with pytest.raises(HipeParsingError, match="BIO"):
        extract(rows)


@pytest.mark.parametrize("meto", [
    ["B-org", "O", "O", "O"],       # overlap from the left
    ["B-org", "I-org", "I-org", "O"],
    ["O", "B-org", "O", "O"],       # only part of location
    ["O", "O", "B-org", "O"],
    ["O", "B-org", "I-org", "I-org"], # continues beyond location
    ["O", "B-org", "B-pers", "O"],  # two contextual spans
])
def test_misalignment(meto):
    lit = ["B-loc", "I-loc", "I-loc", "O"] if meto[0] == "B-org" and meto[1] == "O" else ["O", "B-loc", "I-loc", "O"]
    rows = [(str(i), lit[i], meto[i], "O", "_") for i in range(4)]
    with pytest.raises(HipeParsingError, match="misaligned METO"):
        extract(rows)


def test_partial_and_nested_are_preserved():
    result, = extract([("Berliner", "B-loc", "O", "B-loc", "Partial-0:6|NoSpaceAfter"),
                       ("other", "B-org", "B-org", "B-loc", "_")])
    assert result["mention"] == "Berliner"
    assert result["partial"] == [{"source_token_index": 0, "annotation": "Partial-0:6"}]
    assert result["source_annotations"][0]["NE-NESTED"] == "B-loc"


@pytest.mark.parametrize("separator", ["\n\n", "\n"])
def test_document_boundaries(separator):
    text = HEADER + META + "Paris\tB-loc\tO\tO\tEndOfSentence" + separator
    text += META.replace("doc-1", "doc-2") + "Berlin\tB-loc\tO\tO\t_\n"
    docs = list(parse_documents(StringIO(text)))
    assert [doc.document_id for doc in docs] == ["doc-1", "doc-2"]
    results = [extract_mentions(doc, language="fr", split="dev")[0] for doc in docs]
    assert all(r["source_token_start"] == 0 for r in results)
    assert results[0]["example_id"] != results[1]["example_id"]


def test_missing_columns():
    with pytest.raises(HipeParsingError, match="missing required columns.*NE-COARSE-METO"):
        list(parse_documents(["TOKEN\tNE-COARSE-LIT\tMISC\n"]))


@pytest.mark.parametrize("text,match", [
    (HEADER + "Paris\tB-loc\tO\tO\t_\n", "document_id"),
    (HEADER + META + "Paris\tB-loc\n", "expected 5 columns"),
    (HEADER + META, "without tokens"),
    ("", "missing header"),
])
def test_bad_structure(text, match):
    with pytest.raises(HipeParsingError, match=match):
        list(parse_documents(StringIO(text)))


def test_language_mismatch():
    doc = document([("Paris", "B-loc", "O", "O", "_")])
    with pytest.raises(HipeParsingError, match="language"):
        extract_mentions(doc, language="de", split="dev")


def test_bio_does_not_continue_across_documents():
    text = HEADER + META + "New\tB-loc\tO\tO\t_\n\n"
    text += META.replace("doc-1", "doc-2") + "York\tI-loc\tO\tO\t_\n"
    docs = list(parse_documents(StringIO(text)))
    with pytest.raises(HipeParsingError, match="BIO"):
        extract_mentions(docs[1], language="fr", split="train")


def test_duplicate_document_ids_rejected():
    block = META + "Paris\tB-loc\tO\tO\t_\n"
    with pytest.raises(HipeParsingError, match="duplicate document_id"):
        list(parse_documents(StringIO(HEADER + block + "\n" + block)))


def test_tabbed_metadata_and_hash_token():
    text = HEADER + META.replace("document_id =", "document_id\t=")
    text += "#\tO\tO\tO\t_\nParis\tB-loc\tO\tO\t_\n"
    doc, = parse_documents(StringIO(text))
    result, = extract_mentions(doc, language="fr", split="dev")
    assert result["source_token_start"] == 1
    assert doc.tokens[0]["TOKEN"] == "#"
