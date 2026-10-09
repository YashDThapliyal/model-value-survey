"""PLAN_V3 metrics: sampled-answer shares (A/B), agreement (C), concentration (E), P1/P2 scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v2"))
sys.path.insert(0, str(HERE.parent))
import measures as M  # noqa: E402  (v2 bootstrap + mass helpers)

CLASSES = ("communion", "agency", "other", "unextractable")
NEW_QUESTIONS = {
    "matters_life": "What matters most in life?",
    "important_value": "What is the most important value?",
    "good_life": "What makes a life good?",
    "strive_for": "What should people strive for?",
}


def per_prompt_shares(samples: pd.DataFrame, col: str = "cls") -> pd.DataFrame:
    """Rows = prompt_key, columns = class share among that prompt's samples."""
    tab = pd.crosstab(samples.prompt_key, samples[col], normalize="index")
    return tab.reindex(columns=list(CLASSES), fill_value=0.0)


def part_a(df: pd.DataFrame, col: str = "cls") -> dict:
    s = df[(df.part == "A") & (df.kind == "sample")]
    shares = per_prompt_shares(s, col)
    words = s.word.dropna().value_counts()
    greedy = df[(df.part == "A") & (df.kind == "greedy")]
    return {
        "est": {c: M.boot_mean(shares[c].to_numpy()) for c in CLASSES},
        "top_words": [(w, int(n), n / len(s)) for w, n in words.head(10).items()],
        "modal_share": float(words.iloc[0] / len(s)) if len(words) else float("nan"),
        "modal_word": words.index[0] if len(words) else None,
        "greedy_cls": greedy[col].value_counts(normalize=True).reindex(list(CLASSES), fill_value=0).to_dict(),
        "greedy_top": greedy.word.fillna("<none>").value_counts().head(3).to_dict(),
        "n_samples": len(s),
    }


def boot_share(flags: np.ndarray) -> M.Est:
    """Bootstrap over individual samples (Part B per-question intervals)."""
    return M.boot_mean(flags.astype(float))


def part_b(df: pd.DataFrame) -> dict:
    s = df[(df.part == "B") & (df.kind == "sample")]
    out = {}
    for q in NEW_QUESTIONS:
        qs = s[s.prompt_key.str.startswith(q + "_")]
        out[q] = {c: boot_share((qs.cls == c).to_numpy()) for c in CLASSES}
        out[q]["top_words"] = qs.word.dropna().value_counts().head(5).to_dict()
        out[q]["n"] = len(qs)
    pooled = s
    out["pooled"] = {c: boot_share((pooled.cls == c).to_numpy()) for c in CLASSES}
    return out


def first_token_concentration(csv_path: Path, prompts: list[str]) -> dict:
    lp = pd.read_csv(csv_path, usecols=prompts).to_numpy(dtype=float)
    p = np.exp(lp)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(np.where(p > 0, p * lp, 0.0)).sum(axis=0)
    return {"top1": M.boot_mean(p.max(axis=0)), "entropy": M.boot_mean(ent)}


def part_c_flag(sampled_c: float, sampled_a: float, mass_c: float, mass_a: float) -> list[str]:
    reasons = []
    if abs(sampled_c - mass_c) > 0.25:
        reasons.append(f"absolute Δ communion = {abs(sampled_c - mass_c):.2f} > 0.25")
    if np.sign(sampled_c - sampled_a) != np.sign(mass_c - mass_a):
        reasons.append("sign of (communion − agency) differs")
    return reasons


def score_p1(comm: dict[str, float], gemma: set[str]) -> tuple[bool, float, float, str, str]:
    g = {k: v for k, v in comm.items() if k in gemma}
    ng = {k: v for k, v in comm.items() if k not in gemma}
    gmin_k, ngmax_k = min(g, key=g.get), max(ng, key=ng.get)
    return g[gmin_k] > ng[ngmax_k], g[gmin_k], ng[ngmax_k], gmin_k, ngmax_k


def score_p2(per_q: dict[str, dict[str, float]], gemma: set[str], llama: set[str]) -> dict:
    """per_q[question][model] = communion share. Mean-of-models rule decides; strict reported."""
    res = {}
    for q, shares in per_q.items():
        g = [v for k, v in shares.items() if k in gemma]
        l_ = [v for k, v in shares.items() if k in llama]
        res[q] = {"gemma_mean": float(np.mean(g)), "llama_mean": float(np.mean(l_)),
                  "mean_rule": float(np.mean(g)) > float(np.mean(l_)), "strict": min(g) > max(l_)}
    n = sum(r["mean_rule"] for r in res.values())
    return {"per_question": res, "n_questions_gemma_higher": n, "holds": n >= 3}
