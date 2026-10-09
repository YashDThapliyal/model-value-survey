"""Part D: the authors' Figure 3 MWLR values, and a comparison with results_v2 section 3.

Imports the vendor generate_figure_3.py functions unmodified and calls them exactly as its
main() does (same model list, prompt key, default target tokens). No model is run and
nothing is downloaded (all CSVs are already present; the script checks this first).

Usage: .venv/bin/python scripts/survey/v3/part_d_values.py
Outputs (results_v3/):
  part_d_authors_mwlr.csv      authors' values, all 21 (Llama, Gemma) pairs, as in their main()
  part_d_matched_strings.csv   decoded strings matched per target in the all-10 merge
  part_d_comparison.csv        authors' value vs our results_v2 'authors_top' + decomposition
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "vendor" / "reward_models_inherit_value_biases_from_pretraining"
sys.path[:0] = [str(VENDOR / "figures"), str(VENDOR / "scripts"),
                str(ROOT / "scripts" / "survey" / "v2"), str(ROOT / "scripts" / "survey")]

import generate_figure_3 as g3  # noqa: E402
from fetch_logprobs import safe_name  # noqa: E402

PAPER_DIR = VENDOR / "data" / "logprobs"
LOCAL_DIR = ROOT / "data" / "logprobs"
OUT = ROOT / "results_v3"
TARGETS = ("Freedom", "Love")  # defaults of g3.compute_all_mwlr_comparisons
REF_GEMMA = "google/gemma-2-2b-it"
REF_LLAMA = "meta-llama/Llama-3.2-3B-Instruct"


def require_files(directory: Path, models: list[str]) -> None:
    missing = [m for m in models if not (directory / f"{safe_name(m)}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"missing CSVs in {directory}: {missing} (not downloading)")


def authors_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactly the computational part of g3.main()."""
    require_files(PAPER_DIR, g3.FIGURE_3_MODELS_ALL)
    merged = g3.load_model_logprobs(str(PAPER_DIR), g3.FIGURE_3_MODELS_ALL, g3.PROMPT_KEY)
    print(f"all-10 merge: {len(merged)} common decoded strings")
    res = g3.compute_all_mwlr_comparisons(merged)
    norm = merged["token_decoded"].str.strip().str.lower()
    matched = merged[norm.isin([t.lower() for t in TARGETS])]
    return res.assign(n_common_strings_all10=len(merged)), matched


def series_for(path: Path, prompt: str, keep_default_na: bool) -> pd.Series:
    """Same steps as g3.load_model_logprobs for one file; read_csv NA handling switchable."""
    kw = {} if not keep_default_na else {"keep_default_na": False, "na_values": {prompt: ["", "nan", "NaN"]}}
    df = pd.read_csv(path, usecols=["token_decoded", prompt], **kw)
    df = df.dropna(subset=["token_decoded"])
    df = df.sort_values(prompt, ascending=False).drop_duplicates("token_decoded", keep="first")
    return df.set_index("token_decoded")[prompt]


def pairwise_rule(p_llama: Path, p_gemma: Path, keep_default_na: bool = False) -> dict:
    """Authors' rule with intersection over the two files only (sign: llama minus gemma)."""
    a = series_for(p_llama, g3.PROMPT_KEY, keep_default_na)
    b = series_for(p_gemma, g3.PROMPT_KEY, keep_default_na)
    common = sorted(set(a.index) & set(b.index))
    merged = pd.DataFrame({"token_decoded": common, "L": a.loc[common].values, "G": b.loc[common].values})
    scores = g3.compute_mwlr(merged["L"].values, merged["G"].values)
    norm = merged["token_decoded"].str.strip().str.lower()
    return {t.lower(): (scores[(norm == t.lower()).values].mean() if (norm == t.lower()).any() else np.nan)
            for t in TARGETS}


def load_ours() -> dict:
    tables = pickle.loads((ROOT / "results_v2" / "tables.pkl").read_bytes())
    return {(r["source"], r["model"]): r["mwlr"] for r in tables["rows"]}


def comparison_rows(authors: pd.DataFrame, ours: dict) -> list[dict]:
    """Comparable cells: (Llama X, gemma-2-2b-it) and (Llama-3.2-3B-Instruct, Gemma Y)."""
    paper = lambda m: PAPER_DIR / f"{safe_name(m)}.csv"  # noqa: E731
    local = lambda m: LOCAL_DIR / f"{safe_name(m)}.csv"  # noqa: E731
    cache: dict = {}

    def pw(key, *args):
        if key not in cache:
            cache[key] = pairwise_rule(*args)
        return cache[key]

    rows = []
    for _, a in authors.iterrows():
        llama, gemma = a["llama_model"], a["gemma_model"]
        cells = []
        if gemma == REF_GEMMA:
            cells.append((("paper", llama), REF_GEMMA, 1, "paper Llama vs local gemma-2-2b-it",
                          (paper(llama), local(REF_GEMMA))))
        if llama == REF_LLAMA:
            cells.append((("paper", gemma), REF_LLAMA, -1, "paper Gemma vs local Llama-3.2-3B-Instruct, sign flipped",
                          (local(REF_LLAMA), paper(gemma))))
        if not cells:
            continue
        pp = pw(("pp", llama, gemma, False), paper(llama), paper(gemma), False)
        pp_kna = pw(("pp", llama, gemma, True), paper(llama), paper(gemma), True)
        for our_key, ref, sign, kind, local_pair in cells:
            pl = pw(("pl", kind, llama, gemma, False), *local_pair, False)
            pl_kna = pw(("pl", kind, llama, gemma, True), *local_pair, True)
            for w in ("freedom", "love"):
                our = ours[our_key][(ref, w)]["authors_top"] * sign
                auth = a[f"{w}_mwlr"]
                rows.append({
                    "llama_model": llama, "gemma_model": gemma, "word": w, "our_cell": kind,
                    "authors_fig3": auth, "ours_v2_authors_top": our,
                    "diff_ours_minus_authors": our - auth,
                    "same_sign": bool(np.sign(our) == np.sign(auth)),
                    "pairwise_paper_paper": pp[w], "pairwise_paper_paper_keepNA_false": pp_kna[w],
                    "pairwise_with_local_ref": pl[w], "pairwise_with_local_ref_keepNA_false": pl_kna[w],
                })
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    authors, matched = authors_values()
    authors.to_csv(OUT / "part_d_authors_mwlr.csv", index=False)
    matched.to_csv(OUT / "part_d_matched_strings.csv", index=False)
    comp = pd.DataFrame(comparison_rows(authors, load_ours()))
    comp.to_csv(OUT / "part_d_comparison.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.6f}".format):
        print(authors.to_string(index=False))
        print(matched.to_string(index=False))
        print(comp.drop(columns="our_cell").to_string(index=False))


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
