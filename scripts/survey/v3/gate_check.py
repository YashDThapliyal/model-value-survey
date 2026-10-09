"""Pre-registered gate: gemma-2-2b-it Part A samples -> modal word must be 'love' with share > 0.5.

Exit 0 = pass, 1 = fail. Writes runs/v3/gate.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from answers import ROOT, load_answers

RAW = ROOT / "runs" / "v3" / "raw" / "google--gemma-2-2b-it.jsonl"
THRESHOLD = 0.5


def main() -> None:
    df = load_answers(RAW)
    samples = df[(df.part == "A") & (df.kind == "sample")]
    counts = samples.word.fillna("<unextractable>").value_counts()
    modal, share = counts.index[0], counts.iloc[0] / len(samples)
    passed = modal == "love" and share > THRESHOLD
    greedy = df[(df.part == "A") & (df.kind == "greedy")].word.value_counts().head(5)
    text = (f"# PLAN_V3 gate: gemma-2-2b-it\n\n- Part A samples: {len(samples)} (27 prompts x 20)\n"
            f"- Most common extracted word: `{modal}`, share {share:.3f}; threshold: `love` and > {THRESHOLD}\n"
            f"- **Result: {'PASS' if passed else 'FAIL'}**\n\nTop 10 sampled words:\n\n"
            + "\n".join(f"- `{w}`: {c} ({c / len(samples):.3f})" for w, c in counts.head(10).items())
            + "\n\nGreedy answers (top 5 words over 27 prompts):\n\n"
            + "\n".join(f"- `{w}`: {c}" for w, c in greedy.items()) + "\n")
    out = ROOT / "runs" / "v3" / "gate.md"
    out.write_text(text)
    print(text)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
