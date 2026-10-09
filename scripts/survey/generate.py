"""Generate next-token log-probabilities for one model on MPS, in the authors' CSV format.

Port of vendor/.../scripts/generate_logprobs.py for Apple Silicon. Same prompts, same
tokenization (see prompting.py), same forward pass (use_cache=False, last-position logits
cast to float32, log_softmax), same output columns.

Usage:
  .venv/bin/python scripts/survey/generate.py --model google/gemma-2-2b-it \
      --mode chat --out data/logprobs/google--gemma-2-2b-it.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from prompting import PROMPT_MODES, build_inputs, freeze_template_date, render_prompt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "reward_models_inherit_value_biases_from_pretraining"
PROMPTS_PATH = VENDOR_ROOT / "config" / "prompts.yaml"
DTYPE = torch.bfloat16  # fixed for all models, see runs/decisions.md
DEVICE = "mps"


def load_prompts() -> dict[str, str]:
    return yaml.safe_load(PROMPTS_PATH.read_text())


def load_model(model_id: str) -> tuple[torch.nn.Module, str]:
    """Load with AutoModelForCausalLM; fall back to the multimodal auto class if needed."""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE)
        loader = "AutoModelForCausalLM"
    except (ValueError, KeyError) as exc:
        from transformers import AutoModelForImageTextToText

        print(f"AutoModelForCausalLM failed ({exc}); trying AutoModelForImageTextToText")
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPE)
        loader = "AutoModelForImageTextToText"
    return model.to(DEVICE).eval(), loader


def next_token_logprobs(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> np.ndarray:
    device_inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**device_inputs, use_cache=False).logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.cpu().numpy()


def build_frame(
    tokenizer, prompt_logprobs: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Same construction as the authors: one row per get_vocab() entry, sorted by id.

    Some tokenizers (e.g. Gemma 3: image token id 262144) list ids beyond the model's
    output layer. The model cannot predict them, so those rows are dropped and returned.
    """
    n_logits = next(iter(prompt_logprobs.values())).shape[0]
    vocab = tokenizer.get_vocab().items()
    dropped = sorted(((n, i) for n, i in vocab if i >= n_logits), key=lambda x: x[1])
    vocab_sorted = sorted(((n, i) for n, i in vocab if i < n_logits), key=lambda x: x[1])
    token_names, token_ids = zip(*vocab_sorted)
    token_decoded = tokenizer.batch_decode([[tid] for tid in token_ids], skip_special_tokens=False)
    ids = np.asarray(token_ids)
    columns = {"token_id": token_ids, "token_name": token_names, "token_decoded": token_decoded}
    for key, lp in prompt_logprobs.items():
        if ids.max() >= lp.shape[0]:
            raise ValueError(f"token id {ids.max()} outside logits of size {lp.shape[0]}")
        columns[key] = lp[ids].tolist()
    return pd.DataFrame(columns), dropped


def run(model_id: str, mode: str, out_path: Path) -> dict:
    if mode not in PROMPT_MODES:
        raise ValueError(f"mode must be one of {PROMPT_MODES}")
    freeze_template_date()
    prompts = load_prompts()
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, loader = load_model(model_id)
    load_s = time.time() - t0

    prompt_logprobs: dict[str, np.ndarray] = {}
    n_nonfinite = 0
    for key, text in prompts.items():
        lp = next_token_logprobs(model, build_inputs(tokenizer, text, mode))
        n_nonfinite += int((~np.isfinite(lp)).sum())
        prompt_logprobs[key] = lp
    infer_s = time.time() - t0 - load_s

    df, dropped = build_frame(tokenizer, prompt_logprobs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    meta = {
        "model": model_id,
        "mode": mode,
        "loader": loader,
        "model_class": type(model).__name__,
        "dtype": str(DTYPE),
        "device": DEVICE,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "logits_size": int(next(iter(prompt_logprobs.values())).shape[0]),
        "vocab_rows": len(df),
        "dropped_vocab_ids_outside_logits": dropped,
        "nonfinite_logprobs": n_nonfinite,
        "load_seconds": round(load_s, 1),
        "inference_seconds": round(infer_s, 1),
        "mps_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20),
        "example_prompt": render_prompt(tokenizer, prompts["greatest_ever_one"], mode),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True, choices=PROMPT_MODES)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    meta = run(args.model, args.mode, args.out)
    print(json.dumps({k: v for k, v in meta.items() if k != "example_prompt"}, indent=2))
    if meta["nonfinite_logprobs"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
