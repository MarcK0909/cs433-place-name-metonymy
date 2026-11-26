"""HIPE-only preparation from Group 1, pinned at d5234ba.

AGPL-3.0; see LICENSE and NOTICE.md. No downloads or source edits.
The public prepare_hipe_file adds strict source-target checks before returning
Group 1's exact prepared schema and field ordering.
"""
from pathlib import Path

from .common import RAW, read_hipe, spans
from metonymy.hipe import HipeParsingError, extract_mentions, parse_documents

HIPE_VERSION = "hipe2020-v2.1"  # HIPE-2022 data, v2.1 (test: v2.1-test-all-unmasked)
HIPE_WINDOW = 128  # source tokens kept on each side; models use fewer (default 64)

# Mentions left out of the binary task, with the reason (project decisions,
# also written to manifests/hipe_exclusions.tsv). Empty: see REVIEWED below.
EXCLUDED = {}

# Mentions whose source annotation needed a reviewed decision (kept).
REVIEWED = {
    "fr:GDL-1948-04-15-a-i0035": "malformed IOB2: 'France' B-loc followed by 'moderne' I-org; "
    "the span is repaired to 'France' (I-org -> B-org); its aligned NE-COARSE-METO is O, so literal",
    "de:luxwort-1938-05-20-a-i0023": "malformed IOB2: 'Frankreichs' B-loc followed by I-pers tokens; "
    "the span is repaired to 'Frankreichs' (I-pers -> B-pers); its aligned NE-COARSE-METO is B-pers, so metonymic",
}


def repair_iob2(tags):
    """Make a tag column well-formed IOB2, as the lab's NER pipeline does.

    An I-x that follows O, or follows a mention of another type, starts a new
    mention: it becomes B-x. In HIPE-2020 this happens twice next to a place
    name, both in training documents (see REVIEWED).
    """
    repaired, previous = [], "O"
    for tag in tags:
        if tag.startswith("I-") and (previous == "O" or previous[2:] != tag[2:]):
            tag = "B-" + tag[2:]
        repaired.append(tag)
        previous = tag
    return repaired


def select_mentions(literal, metonymic):
    """The targets of the task in one document, from the two HIPE tag columns.

    Every complete B-loc/I-loc span of the (repaired) literal column is a
    target. Its label comes from the aligned NE-COARSE-METO annotation: O on
    every token = literal; an entity span with exactly the same boundaries =
    metonymic, with that span's type (metonymic-org, metonymic-loc, ...).
    A "_" or a span that only partly overlaps the target stops with an error,
    as the lab's experiment guide asks. This is the only place that decides
    what a target is. Returns (start, end, label, label_class) per target.
    """
    meto_spans = {(start, end): kind for start, end, kind in spans(repair_iob2(metonymic))}
    mentions = []
    for start, end, kind in spans(literal):
        if kind != "loc":
            continue
        tags = metonymic[start:end]
        if any(tag == "_" for tag in tags):
            raise ValueError(f"tokens {start}-{end}: NE-COARSE-METO is '_' (no annotation)")
        if all(tag == "O" for tag in tags):
            mentions.append((start, end, 0, "literal"))
        elif (start, end) in meto_spans:
            mentions.append((start, end, 1, f"metonymic-{meto_spans[start, end]}"))
        else:
            raise ValueError(f"tokens {start}-{end}: metonymy annotation does not align with the place name")
    return mentions



def sentence_numbers(rows, literal):
    """Sentence number of every token. EndOfSentence ends a sentence, except
    inside a mention (next literal tag is I-): HIPE also puts it on
    abbreviation periods such as "Hr ."."""
    numbers, number = [], 0
    for index, row in enumerate(rows):
        numbers.append(number)
        inside_mention = index + 1 < len(rows) and literal[index + 1].startswith("I-")
        if "EndOfSentence" in row[9] and not inside_mention:
            number += 1
    return numbers


def read_hipe_examples(language, split, path=None):
    """Every target place name in one HIPE file, with a window of context.

    Context: up to HIPE_WINDOW source tokens on each side, never crossing the
    document boundary (models use fewer; the shared baseline uses 64). The
    window keeps HIPE's own tokens and spacing (NoSpaceAfter). The id refers
    to the mention by dataset version, language, document ID and original
    token boundaries, as the lab's guide asks; it does not depend on the
    context size or the tokenizer.
    """
    path = path or RAW / f"hipe2020_{language}" / f"HIPE-2022-v2.1-hipe2020-{split}-{language}.tsv"
    examples = []
    for document in read_hipe(path):
        rows = document["rows"]
        raw_literal = [row[1] for row in rows]
        literal = repair_iob2(raw_literal)
        sentence = sentence_numbers(rows, literal)
        try:
            mentions = select_mentions(literal, [row[2] for row in rows])
        except ValueError as error:
            raise ValueError(f"{path.name}, document {document['id']}: {error}") from None
        for start, end, label, label_class in mentions:
            reference = f"{language}:{document['id']}"
            mention_id = f"{HIPE_VERSION}:{language}:{document['id']}:{start}-{end}"
            if mention_id in EXCLUDED:
                continue
            first, last = max(0, start - HIPE_WINDOW), min(len(rows), end + HIPE_WINDOW)
            repaired_next = end < len(rows) and raw_literal[end] != literal[end]
            in_sentence = [i for i in range(first, last) if sentence[start] <= sentence[i] <= sentence[end - 1]]
            examples.append(
                {
                    "id": mention_id,
                    "dataset": f"hipe_{language}",
                    "split": split,
                    "language": language,
                    "document": document["id"],
                    "date": document["date"],
                    "doc_start": start,
                    "doc_end": end,
                    "tokens": [row[0] for row in rows[first:last]],
                    "space_after": ["NoSpaceAfter" not in row[9] for row in rows[first:last]],
                    "start": start - first,
                    "end": end - first,
                    "sentence_start": in_sentence[0] - first,
                    "sentence_end": in_sentence[-1] + 1 - first,
                    "label": label,
                    "label_class": label_class,
                    "partial": any("Partial" in row[9] for row in rows[start:end]),
                    "note": REVIEWED.get(reference, "") if repaired_next else "",
                    "group": document["id"],
                }
            )
    return examples


# Only the two previously inspected source inconsistencies are accepted.
KNOWN_ISSUES = {
    ("fr", "train", "GDL-1948-04-15-a-i0035", "NE-COARSE-LIT", 1462, 1463, "I-org"):
        ("France", ["moderne"]),
    ("de", "train", "luxwort-1938-05-20-a-i0023", "NE-COARSE-LIT", 27, 31, "I-pers"):
        ("Frankreichs", ["größten", "Außenminister", "Deleass", "."]),
}


def prepare_hipe_file(path: str | Path, *, language: str, split: str) -> list[dict]:
    """Prepare approved FR/DE TRAIN/DEV with target/label parity checks.

    Reject unseen malformed annotations rather than letting upstream repair
    silently accept them. Preserve the two agreed targets, all Partial examples,
    source token spellings and spacing. No tokenization or model is involved.
    """
    if language not in ("fr", "de") or split not in ("train", "dev"):
        raise ValueError("This preparation batch supports only French/German TRAIN and DEV")
    path = Path(path)
    expected = {}
    with path.open(encoding="utf-8-sig") as stream:
        for document in parse_documents(stream, source_path=str(path)):
            issues = []
            mentions = extract_mentions(document, language=language, split=split, review_issues=issues)
            for issue in issues:
                key = (language, split, document.document_id, issue["column"],
                       issue["source_token_start"], issue["source_token_end"], issue["tag"])
                actual = (document.tokens[issue["source_token_start"] - 1]["TOKEN"],
                          [row["TOKEN"] for row in issue["source_annotations"]])
                if key not in KNOWN_ISSUES or actual != KNOWN_ISSUES[key]:
                    raise HipeParsingError(f"Unreviewed malformed annotation: {key}; {actual}")
            for mention in mentions:
                key = (document.document_id, mention["source_token_start"], mention["source_token_end"])
                expected[key] = mention["label"]
    prepared = read_hipe_examples(language, split, path)
    observed = {(r["document"], r["doc_start"], r["doc_end"]): r["label"] for r in prepared}
    if len(observed) != len(prepared) or observed != expected:
        raise ValueError(f"{path}: Group 1 preparation disagrees with the validated source mentions")
    return prepared
