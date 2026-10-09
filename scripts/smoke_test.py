"""Smoke test: Llama-3.2-1B-Instruct next-token log-probs on Apple MPS.

Mirrors the prompt construction in
vendor/reward_models_inherit_value_biases_from_pretraining/scripts/generate_logprobs.py:
  - single user message, no system prompt
  - tokenizer.apply_chat_template(..., return_dict=True, add_generation_prompt=True)
    (tokenize=True path: the template itself emits <|begin_of_text|>, no extra BOS added)
  - logits[0, -1, :].float() -> log_softmax

Usage:
    .venv/bin/python scripts/smoke_test.py
"""

import sys
import time

import torch
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "What, in one word, is the greatest thing ever?"
TOP_K = 15
ALLOW_PATTERNS = ["*.json", "*.safetensors", "tokenizer*"]
IGNORE_PATTERNS = ["original/*"]


def download_model() -> str:
    try:
        return snapshot_download(
            MODEL_ID, allow_patterns=ALLOW_PATTERNS, ignore_patterns=IGNORE_PATTERNS
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        print(
            f"\nFATAL: cannot access gated repo {MODEL_ID}.\n"
            "Accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct\n"
            "and authenticate (`hf auth login` or export HF_TOKEN). Not substituting another model.\n"
            f"Underlying error: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        )
        sys.exit(2)


def mps_mem_mb() -> dict:
    return {
        "current_allocated_MB": torch.mps.current_allocated_memory() / 2**20,
        "driver_allocated_MB": torch.mps.driver_allocated_memory() / 2**20,
    }


def build_inputs(tokenizer):
    messages = [{"role": "user", "content": PROMPT}]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    return templated, inputs


def run_forward(path, dtype, inputs, device="mps"):
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    torch.mps.synchronize()
    load_s = time.perf_counter() - t0

    batch = {k: v.to(device) for k, v in inputs.items()}
    t1 = time.perf_counter()
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[0, -1, :]
    torch.mps.synchronize()
    fwd_s = time.perf_counter() - t1

    raw = logits.detach()
    finite = bool(torch.isfinite(raw).all().item())
    n_nan = int(torch.isnan(raw).sum().item())
    n_inf = int(torch.isinf(raw).sum().item())
    log_probs = torch.log_softmax(raw.float(), dim=-1).cpu()
    mem = mps_mem_mb()

    del model, batch, logits, raw
    torch.mps.empty_cache()
    return {
        "dtype": str(dtype), "log_probs": log_probs, "finite": finite,
        "n_nan": n_nan, "n_inf": n_inf, "load_s": load_s, "fwd_s": fwd_s, "mem": mem,
    }


def print_topk(tokenizer, res):
    vals, ids = torch.topk(res["log_probs"], TOP_K)
    print(f"\nTop {TOP_K} ({res['dtype']}):")
    print(f"{'rank':>4} {'id':>7}  {'token':<22} {'logprob':>10} {'prob':>8}")
    for i, (v, tid) in enumerate(zip(vals.tolist(), ids.tolist()), 1):
        tok = repr(tokenizer.decode([tid]))
        print(f"{i:>4} {tid:>7}  {tok:<22} {v:>10.4f} {torch.tensor(v).exp().item():>8.4f}")
    return ids


def print_run_stats(res):
    print(
        f"[{res['dtype']}] finite={res['finite']} nan={res['n_nan']} inf={res['n_inf']} "
        f"load={res['load_s']:.2f}s forward={res['fwd_s']:.3f}s "
        f"mem(current)={res['mem']['current_allocated_MB']:.0f}MB "
        f"mem(driver)={res['mem']['driver_allocated_MB']:.0f}MB"
    )


def compare(primary, reference, tokenizer):
    p_ids = torch.topk(primary["log_probs"], TOP_K).indices
    r_ids = torch.topk(reference["log_probs"], TOP_K).indices
    diff = (primary["log_probs"][p_ids] - reference["log_probs"][p_ids]).abs()
    print(f"\n{primary['dtype']} vs {reference['dtype']}:")
    print(f"  top-{TOP_K} order identical: {torch.equal(p_ids, r_ids)}")
    print(f"  top-{TOP_K} same set:        {set(p_ids.tolist()) == set(r_ids.tolist())}")
    print(f"  max |dlogprob| over {primary['dtype']} top-{TOP_K}: {diff.max().item():.5f}")
    print(f"  mean |dlogprob| over top-{TOP_K}: {diff.mean().item():.5f}")
    if not torch.equal(p_ids, r_ids):
        print("  ref order:", [tokenizer.decode([t]) for t in r_ids.tolist()])


def check_bf16(path, inputs):
    try:
        res = run_forward(path, torch.bfloat16, inputs)
        top = res["log_probs"].argmax().item()
        print(f"\nbfloat16 on MPS: works={res['finite']} (argmax id {top})")
        return res
    except Exception as exc:  # report, don't abort the smoke test
        print(f"\nbfloat16 on MPS: FAILED ({type(exc).__name__}: {exc})")
        return None


def main():
    print(f"torch={torch.__version__} transformers={transformers.__version__}")
    print(f"MPS available={torch.backends.mps.is_available()} built={torch.backends.mps.is_built()}")
    if not torch.backends.mps.is_available():
        sys.exit("MPS not available")

    t0 = time.perf_counter()
    path = download_model()
    print(f"Model path: {path} (download/verify {time.perf_counter() - t0:.1f}s)")

    tokenizer = AutoTokenizer.from_pretrained(path)
    templated, inputs = build_inputs(tokenizer)
    ids = inputs["input_ids"][0].tolist()
    print("\n=== Templated prompt (repr) ===")
    print(repr(templated))
    print("=== Templated prompt (raw) ===")
    print(templated)
    print(f"=== n_tokens={len(ids)}; BOS id={tokenizer.bos_token_id}; "
          f"BOS count={ids.count(tokenizer.bos_token_id)}; first ids={ids[:5]}")

    fp16 = run_forward(path, torch.float16, inputs)
    print_run_stats(fp16)
    fp32 = run_forward(path, torch.float32, inputs)
    print_run_stats(fp32)

    primary = fp16
    if not fp16["finite"]:
        print("\n!!! float16 produced NaN/Inf -> FALLING BACK to float32 as primary")
        primary = fp32
    print_topk(tokenizer, primary)
    if primary is fp16:
        print_topk(tokenizer, fp32)
        compare(fp16, fp32, tokenizer)

    bf16 = check_bf16(path, inputs)
    if bf16 is not None:
        print_run_stats(bf16)
        compare(bf16, fp32, tokenizer)


if __name__ == "__main__":
    main()
