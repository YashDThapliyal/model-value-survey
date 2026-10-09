"""One pass over a model's log-prob CSV -> everything results_v2 needs (small, cached).

Variant rule: a token is a variant of Big Two word w iff its decoded string, with leading
spaces removed, lower-cases to w ("love", "Love", " Love", "LOVE"). Distinct token ids are
distinct events, so their probabilities are summed.
"""

from __future__ import annotations

import pickle
import string
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis_lib as lib  # noqa: E402

MWLR_TARGETS = ("freedom", "love")
TOP_PROMPT = "greatest_ever_one"
# Declared before any v2 numbers were computed. Matched case-insensitively on the token
# with surrounding whitespace removed. User-given: I, The, Subject, **, Wow.
LEAD_IN_WORDS = frozenset({
    "i", "the", "subject", "wow", "a", "an", "it", "its", "it's", "that", "this", "there",
    "in", "as", "what", "my", "well", "oh", "okay", "ok", "hmm", "sure", "here", "answer",
    "so", "if", "to", "of", "for", "we", "you", "is", "and", "but", "honestly", "ah",
})
_PUNCT = set(string.punctuation)


@dataclass(frozen=True)
class Extract:
    variant_mass: pd.DataFrame      # index (Category, word); columns prompts; summed prob
    n_variants: pd.Series           # index (Category, word) -> number of variant tokens
    exact_lp: pd.DataFrame          # authors' lowercase exact-match rows: token_decoded + prompts
    target_rows: pd.DataFrame       # rows whose strip().lower() is a MWLR target (+ prompts)
    class_mass: pd.DataFrame        # index class; columns prompts; prob mass per token class
    top_tokens: pd.DataFrame        # per prompt: top token, prob, class
    n_vocab: int


def token_class(decoded: pd.Series, big2_words: set[str], dict_words: set[str]) -> pd.Series:
    core = decoded.str.strip()
    lower = core.str.lower()
    punct_only = core.map(lambda s: s == "" or all(ch in _PUNCT for ch in s))
    alpha = core.str.isalpha()
    big2 = decoded.str.lstrip(" ").str.lower().isin(big2_words)
    out = np.select(
        [punct_only | lower.isin(LEAD_IN_WORDS), big2,
         alpha & (core.str.len() >= 3) & lower.isin(dict_words), alpha],
        ["lead-in", "big2-variant", "other-word", "fragment"],
        default="other",
    )
    return pd.Series(out, index=decoded.index)


def extract(spec: lib.ModelSpec, big2: pd.DataFrame, prompts: list[str],
            dict_words: set[str]) -> Extract:
    lp = lib.read_logprobs(spec.path, prompts)
    probs = np.exp(lp[prompts].to_numpy(dtype=float))
    norm = lp.token_decoded.str.lstrip(" ").str.lower()
    words = big2.assign(word=big2.token_decoded)

    in_vocab = norm.isin(set(words.word))
    var = pd.DataFrame(probs[in_vocab.to_numpy()], columns=prompts).assign(word=norm[in_vocab].values)
    summed = var.groupby("word")[prompts].sum()
    counts = var.groupby("word").size()
    idx = pd.MultiIndex.from_frame(words[["Category", "word"]])
    variant_mass = summed.reindex(words.word).set_axis(idx).fillna(0.0)
    n_variants = counts.reindex(words.word).set_axis(idx).fillna(0).astype(int)

    exact_lp = lp[lp.token_decoded.isin(set(words.word))][["token_decoded", *prompts]]
    target_rows = lp[lp.token_decoded.str.strip().str.lower().isin(MWLR_TARGETS)][["token_decoded", *prompts]]

    cls = token_class(lp.token_decoded, set(words.word), dict_words)
    class_mass = pd.DataFrame(probs, columns=prompts).groupby(cls.values).sum()
    top_idx = probs.argmax(axis=0)
    top_tokens = pd.DataFrame({
        "prompt": prompts,
        "token": lp.token_decoded.to_numpy()[top_idx],
        "prob": probs[top_idx, np.arange(len(prompts))],
        "class": cls.to_numpy()[top_idx],
    })
    return Extract(variant_mass, n_variants, exact_lp.reset_index(drop=True),
                   target_rows.reset_index(drop=True), class_mass, top_tokens, len(lp))


def cached_extract(spec: lib.ModelSpec, big2: pd.DataFrame, prompts: list[str],
                   dict_words: set[str], cache_dir: Path) -> Extract:
    path = cache_dir / f"{spec.source}--{lib.safe_name(spec.model_id)}.pkl"
    if path.exists() and path.stat().st_mtime > spec.path.stat().st_mtime:
        return pickle.loads(path.read_bytes())
    result = extract(spec, big2, prompts, dict_words)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result
