"""Answer-word extraction and classification, exactly as pre-registered in results_v3.md."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPANDED = (ROOT / "vendor" / "reward_models_inherit_value_biases_from_pretraining"
            / "data" / "corpora" / "dict_big2_expanded.csv")
NOUNS = EXPANDED.with_name("dict_big2_nouns.csv")

MARKUP = "*_#`~>"
QUOTES = "\"“”‘’«»"
LEAD_INS = ("final answer:", "subject:", "answer:", "one word:", "the answer is", "my answer is")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
_STRIP = re.compile(r"^[\s\W_]+")


def extract_word(text: str) -> str | None:
    """Pre-registered rule: strip markup/quotes, drop declared lead-ins, first alphabetic word."""
    s = text.translate({ord(c): " " for c in MARKUP + QUOTES})
    s = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", " ", s)  # apostrophe only at word edges
    s = _STRIP.sub("", s)
    changed = True
    while changed:
        changed = False
        for lead in LEAD_INS:
            if s.lower().startswith(lead):
                s = _STRIP.sub("", s[len(lead):])
                changed = True
    m = WORD_RE.search(s)
    return m.group(0).lower() if m else None


def load_dictionary(path: Path = EXPANDED) -> dict[str, str]:
    d = pd.read_csv(path)
    if "DicTerm" in d.columns:  # expanded dictionary: X marks
        d = d.assign(term=d.DicTerm.str.lower())
        return {**{t: "agency" for t in d.term[d.Agency == "X"]},
                **{t: "communion" for t in d.term[d.Communion == "X"]}}
    return {t.lower(): c.lower() for t, c in zip(d.token_decoded, d.Category)}


def classify(word: str | None, dictionary: dict[str, str]) -> str:
    if word is None or pd.isna(word):  # pandas turns the extractor's None into NaN
        return "unextractable"
    return dictionary.get(word, "other")


def load_answers(raw_path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in raw_path.open()]
    df = pd.DataFrame(rows)
    expanded, nouns = load_dictionary(EXPANDED), load_dictionary(NOUNS)
    words = df.answer_text.map(extract_word)
    return df.assign(word=words, cls=words.map(lambda w: classify(w, expanded)),
                     cls_nouns=words.map(lambda w: classify(w, nouns)))


if __name__ == "__main__":
    tests = {
        "Love": "love", "**Love**": "love", "Subject: Freedom": "freedom", "Answer: \"Joy\"": "joy",
        "The answer is: Family.": "family", "The greatest thing is love": "the", "  \n": None,
        "Final answer: **Kindness**": "kindness", "“Hope”": "hope", "self-love!": "self-love",
        "'Love'": "love", "family's": "family's", "123": None, "Subject: Answer: Peace": "peace",
    }
    for text, want in tests.items():
        got = extract_word(text)
        assert got == want, (text, got, want)
    print("extraction rule tests pass:", len(tests))
