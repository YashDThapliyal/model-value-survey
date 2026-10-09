"""Measures for results_v2: variant-merged mass, valence-controlled gaps, MWLR, ties.

Bootstrap: percentile 95% intervals, B resamples of prompts with replacement, done
independently within each valence (27 prompts each). The 54 prompts come from a factorial
design (adjective x superlative x phrasing) and are not independent draws, so these
intervals describe prompt-to-prompt variability within this prompt set only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from extract import Extract

B = 10_000
SEED = 20261008
CATS = ("Agency", "Communion")


@dataclass(frozen=True)
class Est:
    value: float
    lo: float
    hi: float

    def fmt(self, digits: int = 3) -> str:
        if np.isnan(self.value):
            return "n/a"
        return f"{self.value:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def _idx(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(B, n))


def _ci(point: float, boots: np.ndarray) -> Est:
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return Est(float(point), float(lo), float(hi))


def boot_mean(x: np.ndarray, seed: int = SEED) -> Est:
    rng = np.random.default_rng(seed)
    return _ci(x.mean(), x[_idx(len(x), rng)].mean(axis=1))


def boot_valence_diff(pos: np.ndarray, neg: np.ndarray, stat, seed: int = SEED) -> Est:
    """stat(arr2d) -> per-row statistic; pos/neg are (n_prompts, k) arrays."""
    rng = np.random.default_rng(seed)
    ip, ineg = _idx(len(pos), rng), _idx(len(neg), rng)
    boots = stat(pos[ip]) - stat(neg[ineg])
    return _ci(float(stat(pos[None])[0] - stat(neg[None])[0]), boots)


# ---- 1. variant-merged mass ---------------------------------------------------------
def category_mass(ex: Extract, words: pd.MultiIndex, prompts: list[str]) -> pd.DataFrame:
    """Per prompt: summed variant probability over the given words, per category."""
    vm = ex.variant_mass.loc[ex.variant_mass.index.isin(words), prompts]
    return vm.groupby(level="Category").sum().reindex(list(CATS)).T.fillna(0.0)


def mass_summary(mass: pd.DataFrame, prompts_v: list[str]) -> dict[str, Est]:
    sub = mass.loc[prompts_v]
    a, c = sub["Agency"].to_numpy(), sub["Communion"].to_numpy()
    return {"agency": boot_mean(a), "communion": boot_mean(c), "a_minus_c": boot_mean(a - c)}


def mass_valence_gap(mass: pd.DataFrame, pos: list[str], neg: list[str]) -> Est:
    d = (mass["Agency"] - mass["Communion"]).to_numpy()[:, None]
    pos_v, neg_v = d[[mass.index.get_loc(p) for p in pos]], d[[mass.index.get_loc(p) for p in neg]]
    return boot_valence_diff(pos_v, neg_v, lambda arr: arr[..., 0].mean(axis=-1))


# ---- 2. rank gap (authors' method) ----------------------------------------------------
def per_prompt_median_ranks(ex: Extract, big2: pd.DataFrame, words: set[str],
                            prompts: list[str]) -> pd.DataFrame:
    """Authors' merge-then-rank (method='average') on lowercase exact-match words."""
    w = big2[big2.token_decoded.isin(words)]
    merged = w.merge(ex.exact_lp, on="token_decoded", how="inner")
    ranks = merged[prompts].rank(method="average", ascending=False)
    return ranks.groupby(merged.Category.values).median().T.reindex(columns=list(CATS))


def rank_gap_valence(med: pd.DataFrame, pos: list[str], neg: list[str]) -> tuple[Est, Est, Est]:
    """(positive gap, negative gap, positive minus negative); gap = median_A - median_C."""
    def stat(arr: np.ndarray) -> np.ndarray:
        return np.median(arr[..., 0], axis=-1) - np.median(arr[..., 1], axis=-1)

    p, n = med.loc[pos].to_numpy(), med.loc[neg].to_numpy()
    rng = np.random.default_rng(SEED)
    pos_est = _ci(float(stat(p[None])[0]), stat(p[_idx(len(p), rng)]))
    neg_est = _ci(float(stat(n[None])[0]), stat(n[_idx(len(n), rng)]))
    return pos_est, neg_est, boot_valence_diff(p, n, stat)


# ---- 3. MWLR (Figure 3) ---------------------------------------------------------------
def _mwlr(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * (p1 + p2) * (np.log(p1) - np.log(p2))
    return np.where((p1 > 0) & (p2 > 0), out, np.nan)


def target_mass(ex: Extract, word: str, prompts: list[str]) -> np.ndarray:
    rows = ex.target_rows[ex.target_rows.token_decoded.str.lstrip(" ").str.lower() == word]
    return np.exp(rows[prompts].to_numpy(dtype=float)).sum(axis=0) if len(rows) else np.zeros(len(prompts))


def mwlr_merged(ex: Extract, ref: Extract, word: str, prompts: list[str]) -> np.ndarray:
    """Variant-merged: p(word) = summed probability over its variants, per prompt."""
    return _mwlr(target_mass(ex, word, prompts), target_mass(ref, word, prompts))


def mwlr_authors(ex: Extract, ref: Extract, word: str, prompt: str) -> float:
    """Authors' Figure 3 rule, pairwise: dedupe by decoded string (keep max), intersect
    decoded strings, mean of per-token MWLR over strings with strip().lower() == word."""
    def dedup(e: Extract) -> pd.Series:
        rows = e.target_rows[e.target_rows.token_decoded.str.strip().str.lower() == word]
        return rows.groupby("token_decoded")[prompt].max()

    a, b = dedup(ex), dedup(ref)
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return float("nan")
    return float(np.nanmean(_mwlr(np.exp(a.loc[common].to_numpy()), np.exp(b.loc[common].to_numpy()))))


# ---- 4/5. answer rate and ties --------------------------------------------------------
def class_shares(ex: Extract, prompts_v: list[str]) -> pd.Series:
    return ex.class_mass[prompts_v].mean(axis=1)


def tie_stats(ex: Extract, big2: pd.DataFrame, words: set[str], prompts: list[str]) -> dict:
    w = big2[big2.token_decoded.isin(words)]
    lp = w.merge(ex.exact_lp, on="token_decoded", how="inner")[prompts]
    tied = lp.apply(lambda col: col.duplicated(keep=False).sum())
    distinct = lp.nunique()
    return {
        "n_ranked": len(lp),
        "tied_mean": float(tied.mean()),
        "tied_max": int(tied.max()),
        "tied_share": float(tied.mean() / len(lp)) if len(lp) else float("nan"),
        "distinct_mean": float(distinct.mean()),
        "prompts_with_ties": int((tied > 0).sum()),
    }
