"""Strict HIPE-2022 hipe2020 parsing, separate from prepared model inputs.

Offsets are document-relative, zero-based and end-exclusive. Partial annotations
are retained verbatim, not applied as character slicing: upstream documents known
offset errors. Mention text therefore reconstructs the complete source tokens.
No model context text is constructed and no nested targets are extracted.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import quote

from metonymy.data import LABEL_NAMES

DATASET_VERSION = "v2.1"
SOURCE_REVISION = "2439cb5cfbcd7b65aff6a25b7715398f8f85e034"
REQUIRED_COLUMNS = {"TOKEN", "NE-COARSE-LIT", "NE-COARSE-METO", "MISC"}


class HipeParsingError(ValueError):
    """A source-format or annotation inconsistency requiring review."""


@dataclass
class HipeDocument:
    """Source rows and comments preserved for later inspection."""

    document_id: str
    metadata: dict[str, str]
    tokens: list[dict[str, str]]
    source_lines: list[int]
    comments: list[str]
    source_path: str


def parse_documents(lines: Iterable[str], *, source_path: str = "<stream>") -> Iterator[HipeDocument]:
    """Parse HIPE header, metadata and document blocks without BIO repair.

    Blank lines end documents; an explicit document_id also starts a new block.
    Sentence/layout flags remain in MISC and do not reset document token indices.
    """
    columns: list[str] | None = None
    metadata: dict[str, str] = {}
    tokens: list[dict[str, str]] = []
    source_lines: list[int] = []
    comments: list[str] = []
    seen_ids: set[str] = set()

    def finish() -> HipeDocument:
        document_id = metadata.get("hipe2022:document_id", "")
        if not document_id:
            raise HipeParsingError(f"{source_path}: missing hipe2022:document_id")
        if document_id in seen_ids:
            raise HipeParsingError(f"{source_path}: duplicate document_id {document_id!r}")
        seen_ids.add(document_id)
        return HipeDocument(document_id, dict(metadata), list(tokens), list(source_lines),
                            list(comments), source_path)

    for number, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        if number == 1:
            line = line.removeprefix("\ufeff")
        if not line.strip():
            if tokens:
                yield finish()
                metadata, tokens, source_lines, comments = {}, [], [], []
            continue
        if line.startswith("# ") or (line.startswith("#") and "\t" not in line):
            key, separator, value = line[1:].strip().partition("=")
            key, value = key.strip(), value.strip()
            if key == "hipe2022:document_id" and tokens:
                yield finish()
                metadata, tokens, source_lines, comments = {}, [], [], []
            if tokens:
                raise HipeParsingError(f"{source_path}:{number}: metadata/comment inside token block")
            comments.append(line)
            if separator:
                if key in metadata:
                    raise HipeParsingError(f"{source_path}:{number}: duplicate metadata key {key}")
                metadata[key] = value
            continue
        fields = line.split("\t")
        if columns is None:
            columns = fields
            missing = REQUIRED_COLUMNS - set(columns)
            if missing:
                raise HipeParsingError(f"{source_path}:{number}: missing required columns: {', '.join(sorted(missing))}")
            if len(set(columns)) != len(columns):
                raise HipeParsingError(f"{source_path}:{number}: duplicate column names")
            continue
        if len(fields) != len(columns):
            raise HipeParsingError(f"{source_path}:{number}: expected {len(columns)} columns, got {len(fields)}")
        row = dict(zip(columns, fields))
        if not row["TOKEN"]:
            raise HipeParsingError(f"{source_path}:{number}: empty TOKEN")
        tokens.append(row)
        source_lines.append(number)
    if columns is None:
        raise HipeParsingError(f"{source_path}: missing header and required columns")
    if tokens:
        yield finish()
    elif metadata:
        raise HipeParsingError(f"{source_path}: metadata block without tokens")


def _spans(
    document: HipeDocument, column: str, review_issues: list[dict] | None = None,
) -> list[tuple[int, int, str]]:
    spans = []
    start, entity_type = 0, None
    orphan_issue = None
    for index in range(len(document.tokens) + 1):
        tag = document.tokens[index][column] if index < len(document.tokens) else "O"
        match = re.fullmatch(r"([BI])-([A-Za-z][A-Za-z0-9_.-]*)", tag)
        if tag != "O" and match is None:
            raise HipeParsingError(f"{document.source_path}: document {document.document_id}, "
                                   f"token {index}, {column}: invalid BIO tag {tag!r}")
        prefix, current_type = match.groups() if match else ("O", None)
        if prefix == "I" and current_type != entity_type:
            if review_issues is None:
                raise HipeParsingError(f"{document.source_path}: document {document.document_id}, "
                                       f"token {index}, {column}: malformed BIO continuation {tag!r}")
            # Audit only: retain the preceding typed span, but never invent a B
            # tag or a new entity for an orphan I run. Preserve the run for review.
            if entity_type is not None:
                spans.append((start, index, entity_type))
            entity_type = None
            if orphan_issue is None or orphan_issue["tag"] != tag:
                orphan_issue = {
                    "document_id": document.document_id, "column": column, "tag": tag,
                    "source_token_start": index, "source_token_end": index,
                    "source_line_numbers": [], "source_annotations": [],
                    "reason": "orphan/type-changing I run; not promoted to an entity",
                }
                review_issues.append(orphan_issue)
            orphan_issue["source_token_end"] = index + 1
            orphan_issue["source_line_numbers"].append(document.source_lines[index])
            orphan_issue["source_annotations"].append(dict(document.tokens[index]))
            continue
        orphan_issue = None
        if prefix != "I":
            if entity_type is not None:
                spans.append((start, index, entity_type))
            entity_type = current_type
            start = index
    return spans


def extract_mentions(
    document: HipeDocument, *, language: str, split: str,
    review_issues: list[dict] | None = None,
) -> list[dict]:
    """Extract complete source locations with exactly aligned binary labels.

    A contextual entity of any syntactically valid type (including loc) means
    METONYMIC. Any overlap must be one span with exactly the source boundaries.
    Language and split are explicit provenance, never inferred from filenames.
    Supplying review_issues explicitly enables candidate-only audit extraction:
    malformed I runs are logged and not promoted to entities. Adjacent/overlapping
    candidates carry annotation_issues and require review. Default remains strict;
    invalid tags and METO misalignment always raise, including in audit mode.
    """
    if not language.strip() or not split.strip():
        raise HipeParsingError("language and split must be nonempty")
    for key, expected in (("hipe2022:language", language), ("hipe2022:dataset", "hipe2020")):
        actual = document.metadata.get(key)
        if actual is not None and actual != expected:
            raise HipeParsingError(f"{document.document_id}: {key}={actual!r}, expected {expected!r}")
    local_issues = [] if review_issues is not None else None
    literal = _spans(document, "NE-COARSE-LIT", local_issues)
    contextual = _spans(document, "NE-COARSE-METO", local_issues)
    if review_issues is not None:
        review_issues.extend(local_issues)
    mentions = []
    for start, end, source_type in literal:
        if source_type != "loc":
            continue
        if any(issue["column"] == "NE-COARSE-METO"
               and issue["source_token_start"] < end and issue["source_token_end"] > start
               for issue in (local_issues or [])):
            raise HipeParsingError(f"{document.document_id}: location [{start}:{end}] overlaps malformed METO")
        overlapping = [(a, b, kind) for a, b, kind in contextual if a < end and b > start]
        if overlapping and (len(overlapping) != 1 or overlapping[0][:2] != (start, end)):
            raise HipeParsingError(f"{document.source_path}: document {document.document_id}, "
                                   f"location [{start}:{end}]: misaligned METO spans {overlapping}")
        contextual_type = overlapping[0][2] if overlapping else None
        label = int(contextual_type is not None)
        rows = document.tokens[start:end]
        parts = []
        partial = []
        for index, row in enumerate(rows):
            flags = row["MISC"].split("|")
            parts.append(row["TOKEN"])
            if index < len(rows) - 1 and "NoSpaceAfter" not in flags and "SpaceAfter=No" not in flags:
                parts.append(" ")
            for flag in flags:
                if flag.startswith("Partial"):
                    partial.append({"source_token_index": start + index, "annotation": flag})
        identity = ("hipe2020", DATASET_VERSION, language, split, document.document_id, str(start), str(end))
        mentions.append({
            "example_id": ":".join(quote(part, safe="") for part in identity),
            "dataset": "hipe2020", "dataset_version": DATASET_VERSION,
            "source_revision": SOURCE_REVISION, "language": language, "split": split,
            "document_id": document.document_id, "source_path": document.source_path,
            "source_token_start": start, "source_token_end": end,
            "source_line_numbers": document.source_lines[start:end],
            "mention": "".join(parts), "label": label, "label_name": LABEL_NAMES[label],
            "contextual_type": contextual_type, "partial": partial,
            "source_annotations": [dict(row) for row in rows],
            "document_metadata": dict(document.metadata),
            **({"annotation_issues": [issue for issue in local_issues
                                      if issue["source_token_start"] <= end
                                      and issue["source_token_end"] >= start]}
               if local_issues is not None else {}),
        })
    return mentions


def load_hipe_mentions(path: str | Path, *, language: str, split: str) -> list[dict]:
    """Load a caller-verified v2.1 file; no downloads, filtering or context text."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [mention for document in parse_documents(stream, source_path=str(path))
                for mention in extract_mentions(document, language=language, split=split)]
