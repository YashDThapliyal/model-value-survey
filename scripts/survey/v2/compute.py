"""Compute all results_v2 tables from the saved log-prob CSVs (no model is run).

Usage: .venv/bin/python scripts/survey/v2/compute.py   -> results_v2/tables.pkl + CSVs
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import analysis_lib as lib  # noqa: E402
import measures as M  # noqa: E402
from extract import MWLR_TARGETS, TOP_PROMPT, cached_extract  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "results_v2"
CACHE = ROOT / "runs" / "v2_cache"
REFERENCES = ("google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B-Instruct")  # local copies
LEAD_IN_FLAG = 0.5      # mean lead-in mass over positive prompts above this -> excluded
LOW_MASS_FLAG = 0.05    # mean Big Two variant mass (shared words) below this -> flagged


def load_all(prompts: list[str], big2: pd.DataFrame) -> tuple[list[lib.ModelSpec], dict]:
    local, _ = lib.local_specs(ROOT / "config" / "models.yaml", ROOT / "data" / "logprobs")
    specs = local + lib.paper_specs()
    dict_words = {w.strip().lower() for w in Path("/usr/share/dict/words").read_text().split("\n") if w}
    extracts = {}
    for s in specs:
        print(f"extract {s.key}", flush=True)
        extracts[s.key] = cached_extract(s, big2, prompts, dict_words, CACHE)
    return specs, extracts


def shared_sets(extracts: dict, big2: pd.DataFrame) -> tuple[set[str], pd.MultiIndex]:
    exact = set(big2.token_decoded)
    for ex in extracts.values():
        exact &= set(ex.exact_lp.token_decoded)
    covered = None
    for ex in extracts.values():
        has = set(ex.n_variants[ex.n_variants > 0].index)
        covered = has if covered is None else covered & has
    return exact, pd.MultiIndex.from_tuples(sorted(covered), names=["Category", "word"])


def model_rows(spec, ex, big2, exact_shared, mass_words, prompts, pos, neg, refs) -> dict:
    mass = M.category_mass(ex, mass_words, prompts)
    own_words = ex.n_variants[ex.n_variants > 0].index
    own_mass = M.category_mass(ex, own_words, prompts)
    med = M.per_prompt_median_ranks(ex, big2, exact_shared, prompts)
    rank_pos, rank_neg, rank_diff = M.rank_gap_valence(med, pos, neg)
    shares_pos = M.class_shares(ex, pos)
    top = ex.top_tokens.set_index("prompt").loc[TOP_PROMPT]
    row = {
        "key": spec.key, "source": spec.source, "model": spec.model_id, "label": spec.label,
        "family": spec.family,
        "n_words_variant_own": {c: int((own_words.get_level_values(0) == c).sum()) for c in M.CATS},
        "n_variant_tokens_own": int(ex.n_variants.sum()),
        "mass_pos": M.mass_summary(mass, pos), "mass_neg": M.mass_summary(mass, neg),
        "mass_own_pos": M.mass_summary(own_mass, pos), "mass_own_neg": M.mass_summary(own_mass, neg),
        "mass_valence_gap": M.mass_valence_gap(mass, pos, neg),
        "rank_pos": rank_pos, "rank_neg": rank_neg, "rank_valence_gap": rank_diff,
        "lead_in_pos": float(shares_pos.get("lead-in", 0.0)),
        "big2_pos": float(shares_pos.get("big2-variant", 0.0)),
        "class_shares_pos": shares_pos.to_dict(),
        "top_token": top.token, "top_prob": float(top.prob), "top_class": top["class"],
        "top_tokens_all": ex.top_tokens,
        "ties_own": M.tie_stats(ex, big2, set(ex.exact_lp.token_decoded), prompts),
        "ties_shared": M.tie_stats(ex, big2, exact_shared, prompts),
        "mwlr": {},
    }
    for ref_key, ref in refs.items():
        for w in MWLR_TARGETS:
            per_prompt = M.mwlr_merged(ex, ref, w, prompts)
            pos_vals = per_prompt[[prompts.index(p) for p in pos]]
            row["mwlr"][(ref_key, w)] = {
                "merged_top": float(per_prompt[prompts.index(TOP_PROMPT)]),
                "authors_top": M.mwlr_authors(ex, ref, w, TOP_PROMPT),
                "merged_pos_mean": M.boot_mean(pos_vals) if not pd.isna(pos_vals).any()
                else M.Est(float("nan"), float("nan"), float("nan")),
            }
    row["flag_lead_in"] = row["lead_in_pos"] > LEAD_IN_FLAG
    row["flag_low_mass"] = (row["mass_pos"]["agency"].value + row["mass_pos"]["communion"].value) < LOW_MASS_FLAG
    return row


def main() -> None:
    prompts, big2 = lib.load_prompt_keys(), lib.load_big2()
    pos = [p for p in prompts if lib.is_good(p)]
    neg = [p for p in prompts if not lib.is_good(p)]
    specs, extracts = load_all(prompts, big2)
    exact_shared, mass_words = shared_sets(extracts, big2)
    refs = {r: extracts[f"local:{r}"] for r in REFERENCES}
    rows = [model_rows(s, extracts[s.key], big2, exact_shared, mass_words, prompts, pos, neg, refs)
            for s in specs]
    meta = {
        "n_exact_shared": {c: int(big2[big2.token_decoded.isin(exact_shared)].Category.eq(c).sum()) for c in M.CATS},
        "n_mass_shared": {c: int((mass_words.get_level_values(0) == c).sum()) for c in M.CATS},
        "lead_in_flag": LEAD_IN_FLAG, "low_mass_flag": LOW_MASS_FLAG, "B": M.B, "seed": M.SEED,
        "references": REFERENCES, "n_pos": len(pos), "n_neg": len(neg),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "tables.pkl").write_bytes(pickle.dumps({"rows": rows, "meta": meta}))
    print(meta)


if __name__ == "__main__":
    main()
