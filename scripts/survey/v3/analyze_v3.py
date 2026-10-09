"""Compute PLAN_V3 results and append them to results_v3.md (below the pre-registration).

Usage: .venv/bin/python scripts/survey/v3/analyze_v3.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "v2"):
    sys.path.insert(0, str(p))
import analysis_lib as lib  # noqa: E402
import measures as M  # noqa: E402
import metrics as V  # noqa: E402
from answers import load_answers  # noqa: E402
from charts_v3 import ORDER, SHORT, share_chart  # noqa: E402
from compute import shared_sets  # noqa: E402
from extract import cached_extract  # noqa: E402

ROOT = HERE.parents[2]
RAW = ROOT / "runs" / "v3" / "raw"
OUT = ROOT / "results_v3"
CACHE = ROOT / "runs" / "v2_cache"
GEMMA = {m for fam, ms in ORDER if fam == "Gemma" for m in ms}
LLAMA = {m for fam, ms in ORDER if fam == "Llama" for m in ms}


def first_token(models: list[str]) -> dict[str, dict]:
    """results_v2 section 1 mass (shared variant set over local + paper models) and Part E (b)."""
    prompts, big2 = lib.load_prompt_keys(), lib.load_big2()
    pos = [p for p in prompts if lib.is_good(p)]
    local, _ = lib.local_specs(ROOT / "config" / "models.yaml", ROOT / "data" / "logprobs")
    specs = local + lib.paper_specs()
    words = {w.strip().lower() for w in Path("/usr/share/dict/words").read_text().split("\n") if w}
    ex = {s.key: cached_extract(s, big2, prompts, words, CACHE) for s in specs}
    _, mass_words = shared_sets(ex, big2)
    out = {}
    for s in local:
        if s.model_id not in models:
            continue
        mass = M.category_mass(ex[s.key], mass_words, prompts).loc[pos]
        share = mass.Communion / (mass.Agency + mass.Communion)
        out[s.model_id] = {
            "mass_c": M.boot_mean(mass.Communion.to_numpy()), "mass_a": M.boot_mean(mass.Agency.to_numpy()),
            "comm_share_of_big2": M.boot_mean(share.fillna(0).to_numpy()),
            **V.first_token_concentration(s.path, pos),
        }
    return out, {c: int((mass_words.get_level_values(0) == c).sum()) for c in M.CATS}


def main() -> None:
    config = {m["id"]: m for m in yaml.safe_load((ROOT / "config" / "models.yaml").read_text())["models"]}
    answers = {}
    for path in sorted(RAW.glob("*.jsonl")):
        df = load_answers(path)
        answers[df.model.iloc[0]] = df
    models = list(answers)
    A = {m: V.part_a(answers[m]) for m in models}
    A_nouns = {m: V.part_a(answers[m], col="cls_nouns") for m in models}
    B = {m: V.part_b(answers[m]) for m in models}
    ft, n_shared = first_token(models)
    comm = {m: A[m]["est"]["communion"].value for m in models}
    p1 = V.score_p1(comm, GEMMA & set(models))
    per_q = {q: {m: B[m][q]["communion"].value for m in models} for q in V.NEW_QUESTIONS}
    p2 = V.score_p2(per_q, GEMMA & set(models), LLAMA & set(models))
    OUT.mkdir(exist_ok=True)
    share_chart({m: A[m]["est"] for m in models},
                "Which value do models name as \"the greatest thing\"?",
                "Original paper prompts (27 positive phrasings). Answer word classified with the Big Two dictionary.",
                OUT / "answer_shares_original.png")
    for q, text in V.NEW_QUESTIONS.items():
        share_chart({m: B[m][q] for m in models}, f"\"{text}\" (new question, not from the paper)",
                    "3 phrasings x 20 samples per model. Answer word classified with the Big Two dictionary.",
                    OUT / f"answer_shares_{q}.png")
    (OUT / "v3_tables.pkl").write_bytes(pickle.dumps(
        {"A": A, "A_nouns": A_nouns, "B": B, "ft": ft, "p1": p1, "p2": p2, "n_shared": n_shared,
         "config": config, "models": models}))
    from report_v3 import write_results  # noqa: E402  (kept separate for file size)
    write_results(ROOT / "results_v3.md", A, A_nouns, B, ft, p1, p2, n_shared, models, SHORT)


if __name__ == "__main__":
    main()
