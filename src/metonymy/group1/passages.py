"""The passage text that the models read, with the target place name located.

The text is rebuilt from HIPE's own tokens and spacing (space_after), so all
groups see the same text. The target can be marked:
    BRACKETS  "Le gouvernement de [[Paris]] a rejeté la proposition."
    TARGET    "Le gouvernement de [TARGET]Paris[/TARGET] a rejeté ..."  (the
              lab's shared LLM prompt)
    None      plain text; the encoder then finds the target by its position
              (the lab's Option A and B)
"""
BRACKETS = ("[[", "]]")
TARGET = ("[TARGET]", "[/TARGET]")


def passage(example, max_words=None, markers=BRACKETS, sentence_only=False):
    """Build the passage of one mention.

    Tokens are joined with a space, except after a token that has no space
    after it in the newspaper (field space_after). max_words keeps at most this
    many tokens on each side of the target (None keeps the whole prepared
    window); the target itself is always kept whole. sentence_only keeps only
    the target's sentence (fields sentence_start/sentence_end, when present).

    Returns a dict: text, name_start and name_end (character span of the place
    name, without markers) and marker_start (position of the opening marker,
    or of the name when there are no markers).
    """
    tokens, start, end = example["tokens"], example["start"], example["end"]
    space_after = example.get("space_after") or [True] * len(tokens)
    low, high = 0, len(tokens)
    if sentence_only and "sentence_start" in example:
        low, high = example["sentence_start"], example["sentence_end"]
    first = low if max_words is None else max(low, start - max_words)
    last = high if max_words is None else min(high, end + max_words)
    opening, closing = markers or ("", "")
    text = ""
    for position in range(first, last):
        if position == start:
            marker_start = len(text)
            text += opening
            name_start = len(text)
        text += tokens[position]
        if position == end - 1:
            name_end = len(text)
            text += closing
        if position < last - 1 and space_after[position]:
            text += " "
    return {"text": text, "name_start": name_start, "name_end": name_end, "marker_start": marker_start}
