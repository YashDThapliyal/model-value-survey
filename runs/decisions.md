# Fixed run decisions (recorded 2026-10-08, BEFORE Gate C was run)

These were fixed before any validation numbers existed, so they could not be tuned to pass a gate.

## Precision: bfloat16 for every model
- The authors generated every one of their 12 files in bfloat16 (vendor `config/models.yaml`).
  Gate C must follow their setup exactly, and PLAN.md requires one precision for all
  models, so the survey uses bfloat16 too.
- Gate B showed bfloat16 is finite on MPS (no NaN/Inf).
- Known trade-off: on Llama-3.2-1B, float16 tracked float32 more closely than bfloat16
  (max |Δlogprob| over top-15: fp16 0.012, bf16 0.115). float16 was rejected because it
  would diverge from the authors' files, and Gemma 2/3 are known to overflow in float16.
- Logits are cast to float32 before log_softmax (same as the authors).

## Date injected by chat templates: 26 Jul 2024
- Applied by freezing the `datetime` that transformers' `strftime_now` template global
  reads, so every template that injects a date sees the same one.
- "26 Jul 2024" is the hard-coded fallback in Meta's Llama 3.x chat template (the date
  Llama renders when no clock is available), so it is a non-arbitrary choice.
- Affected templates: Llama 3.2 (both), SmolLM3 (if it injects a date). Others are
  checked and listed in results.md.
- The authors' own runs used their (unknown) run date, so Llama prompts cannot match
  theirs token-for-token. Gate C reports this as a known difference.

## Prompt modes
- Instruct models: authors' setup, `apply_chat_template([user msg], add_generation_prompt=True)`,
  no system prompt.
- Qwen3, SmolLM3: same, plus `enable_thinking=False`.
- DeepSeek-R1-Distill-Qwen-1.5B: chat template, then the assistant turn is prefilled with an
  empty thinking block (`<think>\n\n</think>\n\n`) so the next token is the answer.
  Results are labelled "empty-think workaround".
- Qwen2.5-Math-1.5B (base model): raw prompt, no template, the authors' "pretrained" path.

## Cache safety
- `runs/protected_cache_entries.txt` lists every HF cache entry that existed before this
  project. The survey deletes only the weights of the model it just processed, and never
  any entry on that list. (Llama-3.2-1B-Instruct was downloaded by this project's smoke
  test, so it is not protected.)

## Attention implementation: library default (as the authors' code)
- The authors' script never sets `attn_implementation`, so it used the transformers default.
  The survey also leaves it unset. In transformers 5.19 that is SDPA, which does NOT apply
  Gemma 2's `attn_logit_softcapping` (only the eager path does). Decided before Gate C;
  listed as a caveat. The final-logit softcapping is applied on every path.

## Device and loading
- MPS, batch size 1, `use_cache=False`, `inference_mode`, one forward pass per prompt
  (same as the authors). The model is loaded with `dtype=bfloat16` then `.to("mps")`, not
  `device_map="auto"`. CUDA determinism flags do not apply on MPS.
- Output CSVs use the authors' exact format: token_id, token_name, token_decoded, then
  the 54 prompt columns in `prompts.yaml` order, one row per `tokenizer.get_vocab()`
  entry, sorted by id.
