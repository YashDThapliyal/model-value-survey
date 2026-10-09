"""PLAN_V3 Parts A/B: greedy + 20 temperature-1.0 samples per prompt, raw outputs saved.

Prompts: the paper's 27 positive prompts (Part A) + 12 new prompts (Part B, config/prompts_v3_new.yaml).
Prompting, dtype, device and date are identical to the survey (scripts/survey/prompting.py).

Usage: .venv/bin/python scripts/survey/v3/sample.py --model ID --mode MODE --out runs/v3/raw/<safe>.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, GenerationConfig

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
from generate import DEVICE, load_model, load_prompts  # noqa: E402
from prompting import PROMPT_MODES, build_inputs, freeze_template_date  # noqa: E402

ROOT = SURVEY.parents[1]
NEW_PROMPTS = ROOT / "config" / "prompts_v3_new.yaml"
SEED = 20261008
N_SAMPLES = 20
MAX_NEW = 12
MAX_NEW_THINKING = 400
POSITIVE_WORDS = ("greatest", "best", "good")  # authors' is_good rule
THINK_CLOSE = "</think>"


def prompt_set() -> list[tuple[str, str, str]]:
    """(part, key, text): 27 original positive prompts then the 12 new ones."""
    original = [("A", k, v) for k, v in load_prompts().items() if any(w in k for w in POSITIVE_WORDS)]
    new = [("B", k, v) for k, v in yaml.safe_load(NEW_PROMPTS.read_text()).items()]
    return original + new


def gen_config(model, sample: bool, max_new: int) -> GenerationConfig:
    """Fresh config: no inherited top_k/top_p/temperature/repetition_penalty from the repo."""
    base = model.generation_config
    eos = base.eos_token_id if base.eos_token_id is not None else model.config.eos_token_id
    pad = base.pad_token_id if base.pad_token_id is not None else (eos[0] if isinstance(eos, list) else eos)
    common = dict(max_new_tokens=max_new, eos_token_id=eos, pad_token_id=pad, repetition_penalty=1.0)
    if sample:
        return GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                num_return_sequences=N_SAMPLES, **common)
    return GenerationConfig(do_sample=False, num_beams=1, **common)


def answer_text(tokenizer, ids: list[int], thinking: bool) -> tuple[str, str]:
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    if not thinking:
        return raw, tokenizer.decode(ids, skip_special_tokens=True)
    if THINK_CLOSE not in raw:
        return raw, ""  # thinking not finished -> unextractable
    tail = raw.rsplit(THINK_CLOSE, 1)[1]
    for tok in tokenizer.all_special_tokens:
        tail = tail.replace(tok, "")
    return raw, tail


def run(model_id: str, mode: str, out: Path) -> dict:
    if mode not in (*PROMPT_MODES, "chat_thinking"):
        raise ValueError(mode)
    thinking = mode == "chat_thinking"
    freeze_template_date()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, loader = load_model(model_id)
    max_new = MAX_NEW_THINKING if thinking else MAX_NEW
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    with out.open("w") as fh:
        for idx, (part, key, text) in enumerate(prompt_set()):
            inputs = {k: v.to(DEVICE) for k, v in build_inputs(tokenizer, text, "chat" if thinking else mode).items()}
            n_in = inputs["input_ids"].shape[1]
            runs = [("greedy", gen_config(model, False, max_new))]
            runs.append(("sample", gen_config(model, True, max_new)))
            for kind, cfg in runs:
                if kind == "sample":
                    torch.manual_seed(SEED + idx)
                with torch.inference_mode():
                    seqs = model.generate(**inputs, generation_config=cfg)
                for i, seq in enumerate(seqs):
                    ids = seq[n_in:].tolist()
                    raw, ans = answer_text(tokenizer, ids, thinking)
                    fh.write(json.dumps({"model": model_id, "part": part, "prompt_key": key, "prompt": text,
                                         "kind": kind, "sample_index": i, "n_new_tokens": len(ids),
                                         "raw": raw, "answer_text": ans}) + "\n")
                    n += 1
            print(f"[{time.time() - t0:7.1f}s] {idx + 1}/39 {key}", flush=True)
    return {"model": model_id, "mode": mode, "loader": loader, "generations": n,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True, choices=(*PROMPT_MODES, "chat_thinking"))
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    meta = run(a.model, a.mode, a.out)
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
