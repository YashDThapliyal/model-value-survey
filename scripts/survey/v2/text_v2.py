"""Fixed prose blocks for results_v2.md (kept out of report.py to keep that file small)."""

METHOD = """Analysis of the saved next-token log-probability files only: 13 local runs in `data/logprobs/` plus the
authors' 12 published files. No model was downloaded or run. Code: `scripts/survey/v2/` (`compute.py`, then
`report.py`). Numbers are reported, not interpreted.

- **Variants:** a token is a variant of Big Two word *w* if its decoded string, with leading spaces removed,
  lower-cases to *w* (`love`, `Love`, ` Love`, `LOVE`). Only single tokens count. Distinct token ids are
  distinct outcomes, so their probabilities are summed.
- **Bootstrap:** 10,000 resamples of prompts with replacement, done separately within each valence (27 positive,
  27 negative prompts), with percentile 95% intervals and seed 20261008. The prompts come from a factorial design
  (adjective × superlative × phrasing), not independent draws, so the intervals describe variation across
  this prompt set only.
- **Valence:** the authors' rule. A prompt is positive if its key contains greatest, best or good.
- **"(ours)" / "(paper)":** whether the row comes from our local MPS run or from the authors' published file."""

CAVEATS = """- **Thinking workaround:** DeepSeek-R1-Distill-Qwen-1.5B is prompted with a prefilled empty `<think>` block. Even so,
  97% of its positive-prompt mass falls on lead-ins (mostly `I`).
- **Thinking disabled:** Qwen3-1.7B, Qwen3-4B and SmolLM3-3B ran with `enable_thinking=False`.
- **Base models:** Qwen2.5-Math-1.5B (ours) and the paper's pretrained Gemma 2 2B and Llama 3.2 3B received raw prompts,
  so their next token is a continuation (`\\n\\n`, ` The`), not an answer.
- **Multimodal models:** gemma-3-4b-it and gemma-4-E2B-it were run through their full multimodal model classes, with text input only.
- **Tokenizer differences:**
  - The number of Big Two words with a single-token variant, and the number of variant tokens, differ by tokenizer. The
    shared word sets put every model on the same words. The own-coverage tables do not.
  - Words with no single-token variant in a model contribute zero mass for that model.
- **Date:** Llama 3.2 and SmolLM3 templates in our runs used the fixed date 26 Jul 2024. The paper's runs used the authors' own (unknown) run date.
- **Precision:** all local runs used bfloat16, as the authors did. The coarse grid of bf16 log-probs is the source of the ties in section 5.
- **Agency mass is small everywhere:** on positive prompts it is below 0.04 for every model, so (A − C) is driven mostly by the Communion term.
- **MWLR reference sets:**
  - The authors' Figure 3 intersects decoded strings across all 10 of their models at once; here the intersection is
    pairwise. Part D of results_v3 (`results_v3/part_d.md`) shows this makes no difference for the Freedom/Love cells:
    the all-10 and pairwise values are identical.
  - The remaining differences from their Figure 3 come from our references being local MPS re-runs. Our Llama-3.2-3B-Instruct
    file has lower log-probs on several Freedom/Love strings than the authors' file.
  - MWLR is undefined (n/a) when either side has zero probability on all variants.
- **Lead-in set:** declared before any v2 number was computed. The `other-word` vs `fragment` split relies on
  `/usr/share/dict/words`, which contains some fragments (e.g. `creat`). It is descriptive only and is not used for any flag."""
