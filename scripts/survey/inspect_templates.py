"""Render the 'greatest thing ever' prompt for every planned model (tokenizer files only).

Usage: .venv/bin/python scripts/survey/inspect_templates.py [--out runs/templates.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prompting import build_inputs, freeze_template_date, render_prompt  # noqa: E402

PROMPT = "What, in one word, is the greatest thing ever?"


def describe(model_id: str, mode: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_id)
    text = render_prompt(tok, PROMPT, mode)
    ids = build_inputs(tok, PROMPT, mode)["input_ids"][0].tolist()
    bos = tok.bos_token_id
    bos_count = ids.count(bos) if bos is not None else 0
    has_date = any(s in text for s in ("2024", "2025", "2026"))
    return (
        f"## {model_id} (mode={mode})\n\n"
        f"- tokens: {len(ids)}, BOS id {bos}, BOS count {bos_count}, "
        f"date in prompt: {has_date}, last 5 ids: {ids[-5:]}\n"
        f"- vocab size (get_vocab): {len(tok.get_vocab())}\n\n"
        f"```\n{text}\n```\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/models.yaml")
    parser.add_argument("--out", default="runs/templates.md")
    args = parser.parse_args()
    freeze_template_date()
    models = yaml.safe_load(Path(args.config).read_text())["models"]
    parts = ["# Rendered prompts (fixed date 26 Jul 2024)\n"]
    for m in models:
        if m.get("status") == "skipped":
            continue
        try:
            parts.append(describe(m["id"], m["prompt_mode"]))
        except Exception as exc:  # report and continue; one bad tokenizer must not hide others
            parts.append(f"## {m['id']}\n\nERROR: {type(exc).__name__}: {exc}\n")
    Path(args.out).write_text("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
