"""Big Two (Agency / Communion) analysis of next-token log-prob CSVs.

Compares our locally generated models (config/models.yaml, data/logprobs/) with the
12 reference models from Christian et al. (vendor repo), using the authors' word
matching and ranking rules (see analysis_lib.py).

Usage:
    .venv/bin/python scripts/survey/analyze.py                 # default paths
    .venv/bin/python scripts/survey/analyze.py --local-dir X   # other local CSV dir
    .venv/bin/python scripts/survey/analyze.py --no-fetch      # don't download paper CSVs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_lib as lib  # noqa: E402
import analysis_report as report  # noqa: E402

TOP_PROMPT = "greatest_ever_one"
TOP_N = 15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--local-dir", type=Path, default=lib.REPO_ROOT / "data" / "logprobs")
    p.add_argument("--config", type=Path, default=lib.REPO_ROOT / "config" / "models.yaml")
    p.add_argument("--out-dir", type=Path, default=lib.REPO_ROOT / "results")
    p.add_argument("--no-fetch", action="store_true",
                   help="do not download missing paper CSVs via the vendor helper")
    p.add_argument("--allow-missing-meta", action="store_true",
                   help="use local CSVs even without a sibling .meta.json")
    return p.parse_args()


def fetch_paper(specs: list[lib.ModelSpec]) -> None:
    sys.path.insert(0, str(lib.VENDOR_ROOT / "scripts"))
    from fetch_logprobs import fetch_missing_logprobs
    fetch_missing_logprobs([s.model_id for s in specs], lib.VENDOR_LOGPROBS)


def shared_words(big2: pd.DataFrame, vocabs: dict[str, pd.Series]) -> pd.DataFrame:
    mask = pd.Series(True, index=big2.index)
    for vocab in vocabs.values():
        mask = mask & big2.token_decoded.isin(vocab)
    return big2[mask]


def analyse_model(spec: lib.ModelSpec, big2: pd.DataFrame, shared: pd.DataFrame,
                  prompts: list[str]) -> dict:
    """Load one model once; return own/shared summaries, duplicates, top/bottom."""
    lp = lib.read_logprobs(spec.path, prompts)
    own, dups = lib.model_result(spec, lib.words_in_vocab(big2, lp.token_decoded), lp, prompts)
    shr, _ = lib.model_result(spec, shared, lp, prompts)
    result = {"own": own, "shared": shr, "dups": dups, "n_vocab": len(lp),
              "n_nonfinite": int((~np.isfinite(lp[prompts].to_numpy(dtype=float))).sum())}
    if spec.source == "local":
        result["top"], result["bottom"] = lib.top_bottom(lp, TOP_PROMPT, TOP_N)
    return result


def write_outputs(out_dir: Path, specs: list[lib.ModelSpec], results: dict[str, dict],
                  shared: pd.DataFrame) -> dict:
    own = pd.concat([results[s.key]["own"] for s in specs], ignore_index=True)
    shr = pd.concat([results[s.key]["shared"] for s in specs], ignore_index=True)
    own.to_csv(out_dir / "per_model_own_vocab.csv", index=False)
    shr.to_csv(out_dir / "per_model_shared_vocab.csv", index=False)

    n_shared = shared.Category.value_counts().reindex(lib.CATEGORIES, fill_value=0).to_dict()
    chart = report.plot_gap(shr[shr.valence == "positive"], n_shared, out_dir / "gap_positive")

    local = [s for s in specs if s.source == "local"]
    md = [f"# Top / bottom {TOP_N} tokens, prompt `{TOP_PROMPT}` (full vocab, local models)", ""]
    md += [report.top_bottom_markdown(s.label, s.model_id, TOP_PROMPT, results[s.key]["n_vocab"],
                                      results[s.key]["top"], results[s.key]["bottom"])
           for s in local]
    if not local:
        md.append("_No local models available yet._")
    (out_dir / "top_bottom_greatest_ever_one.md").write_text("\n".join(md))
    return {"n_shared": n_shared, "chart": [str(p) for p in chart]}


def build_summary(specs: list[lib.ModelSpec], results: dict[str, dict], notes: list[str],
                  extra: dict, prompts: list[str]) -> dict:
    def model_entry(s: lib.ModelSpec) -> dict:
        r = results[s.key]
        own_row = r["own"].iloc[0]
        pos = r["shared"].set_index("valence").loc["positive"]
        return {
            "source": s.source, "model_id": s.model_id, "label": s.label,
            "family": s.family, "prompt_mode": s.prompt_mode, "vocab_rows": r["n_vocab"],
            "nonfinite_logprobs": r["n_nonfinite"],
            "n_agency_words_own": int(own_row.n_agency_words),
            "n_communion_words_own": int(own_row.n_communion_words),
            "duplicate_words": r["dups"].to_dict("records"),
            "shared_positive_gap_median": float(pos.gap_median),
            "meta": {k: v for k, v in s.meta.items() if k != "example_prompt"},
        }
    return {
        "n_prompts": len(prompts),
        "n_positive_prompts": sum(lib.is_good(k) for k in prompts),
        "models_included": [model_entry(s) for s in specs],
        "local_skipped": notes,
        "shared_word_counts": extra["n_shared"],
        "outputs": ["per_model_own_vocab.csv", "per_model_shared_vocab.csv",
                    *[Path(p).name for p in extra["chart"]],
                    "top_bottom_greatest_ever_one.md", "analysis_summary.json"],
    }


def print_summary(summary: dict, own: pd.DataFrame) -> None:
    print(f"\nShared words: {summary['shared_word_counts']}")
    for note in summary["local_skipped"]:
        print(f"  skipped local {note}")
    cols = ["source", "label", "valence", "n_agency_words", "n_communion_words",
            "agency_median_rank", "communion_median_rank", "gap_median", "n_duplicate_words"]
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", "{:.1f}".format):
        print("\nOwn-vocab medians:")
        print(own[cols].to_string(index=False))
        print("\nShared-vocab positive gap (median):")
        for m in summary["models_included"]:
            print(f"  {m['source']:5s} {m['label']:45s} {m['shared_positive_gap_median']:7.2f}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = lib.load_prompt_keys()
    big2 = lib.load_big2()
    paper = lib.paper_specs()
    if not args.no_fetch:
        fetch_paper(paper)
    paper = [s for s in paper if s.path.exists()]
    local, notes = lib.local_specs(args.config, args.local_dir, not args.allow_missing_meta)
    specs = [*local, *paper]
    print(f"Models: {len(local)} local, {len(paper)} paper")

    shared = shared_words(big2, {s.key: lib.read_vocab(s.path) for s in specs})
    results = {}
    for s in specs:
        print(f"  analysing {s.key}")
        results[s.key] = analyse_model(s, big2, shared, prompts)
    extra = write_outputs(args.out_dir, specs, results, shared)
    summary = build_summary(specs, results, notes, extra, prompts)
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print_summary(summary, pd.read_csv(args.out_dir / "per_model_own_vocab.csv"))


if __name__ == "__main__":
    main()
