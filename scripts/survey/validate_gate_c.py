"""Gate C: compare locally generated log-probs against the authors' precomputed files.

Per model:
  - Spearman correlation (local vs authors) over the Big Two nouns present in the vocabulary
    (authors' rule: dictionary word == a token's `token_decoded`), for each of the 54 prompts.
  - Top-15 token overlap (full vocabulary) on the "greatest_ever_one" prompt.
Gate passes for a model iff the median Spearman over the 54 prompts is >= 0.9.

Usage: .venv/bin/python scripts/survey/validate_gate_c.py --models A B --out runs/gate_c.md
Exit code 0 = all pass, 1 = at least one model fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "reward_models_inherit_value_biases_from_pretraining"
AUTHORS_DIR = VENDOR_ROOT / "data" / "logprobs"
LOCAL_DIR = PROJECT_ROOT / "data" / "logprobs"
BIG2_PATH = VENDOR_ROOT / "data" / "corpora" / "dict_big2_nouns.csv"
PROMPTS_PATH = VENDOR_ROOT / "config" / "prompts.yaml"
THRESHOLD = 0.9
TOP_K = 15
TOP_PROMPT = "greatest_ever_one"


@dataclass(frozen=True)
class GateResult:
    model: str
    n_words: int
    per_prompt: pd.Series
    top_local: list[str]
    top_authors: list[str]
    decoded_mismatches: int

    @property
    def median(self) -> float:
        return float(self.per_prompt.median())

    @property
    def passed(self) -> bool:
        return self.median >= THRESHOLD

    @property
    def top_overlap(self) -> int:
        return len(set(self.top_local) & set(self.top_authors))


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def top_tokens(df: pd.DataFrame, col: str, k: int = TOP_K) -> list[str]:
    return df.nlargest(k, col)["token_decoded"].astype(str).tolist()


def compare(model_id: str, prompts: list[str], big2: pd.DataFrame) -> GateResult:
    cols = ["token_id", "token_decoded", *prompts]
    local = pd.read_csv(LOCAL_DIR / f"{safe_name(model_id)}.csv", usecols=cols, keep_default_na=False)
    authors = pd.read_csv(AUTHORS_DIR / f"{safe_name(model_id)}.csv", usecols=cols, keep_default_na=False)
    merged = local.merge(authors, on="token_id", suffixes=("_local", "_authors"))
    mismatches = int((merged["token_decoded_local"] != merged["token_decoded_authors"]).sum())
    in_dict = merged[merged["token_decoded_authors"].isin(big2["token_decoded"])]
    rho = {
        p: spearmanr(in_dict[f"{p}_local"], in_dict[f"{p}_authors"]).statistic for p in prompts
    }
    return GateResult(
        model=model_id,
        n_words=in_dict["token_decoded_authors"].nunique(),
        per_prompt=pd.Series(rho),
        top_local=top_tokens(local, TOP_PROMPT),
        top_authors=top_tokens(authors, TOP_PROMPT),
        decoded_mismatches=mismatches,
    )


def report(results: list[GateResult]) -> str:
    lines = ["# Gate C: local (MPS, bf16) vs authors' precomputed log-probs", ""]
    lines += [f"Pass rule: median Spearman over 54 prompts >= {THRESHOLD}.", ""]
    lines += ["| Model | Big Two words | Median ρ | Min ρ | Max ρ | Top-15 overlap | Result |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.per_prompt
        lines.append(
            f"| `{r.model}` | {r.n_words} | {r.median:.4f} | {s.min():.4f} | {s.max():.4f} "
            f"| {r.top_overlap}/{TOP_K} | {'PASS' if r.passed else 'FAIL'} |"
        )
    for r in results:
        lines += ["", f"## {r.model}", ""]
        lines.append(f"- token_decoded mismatches between local and authors' files: {r.decoded_mismatches}")
        lines.append(f"- 5 lowest-ρ prompts: {r.per_prompt.nsmallest(5).round(4).to_dict()}")
        lines += ["", "| Rank | Local top-15 | Authors' top-15 |", "|---|---|---|"]
        for i, (a, b) in enumerate(zip(r.top_local, r.top_authors), 1):
            lines.append(f"| {i} | `{a!r}` | `{b!r}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "runs" / "gate_c.md")
    args = parser.parse_args()
    prompts = list(yaml.safe_load(PROMPTS_PATH.read_text()))
    big2 = pd.read_csv(BIG2_PATH)
    results = [compare(m, prompts, big2) for m in args.models]
    text = report(results)
    args.out.write_text(text)
    print(text)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
