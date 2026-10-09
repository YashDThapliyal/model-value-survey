I'm logged in to Hugging Face now and have access to the Llama and Gemma models.
Decisions on your open questions are below, then the plan for tonight. Run it
unattended, but stop at any gate that fails and write down why. A wrong pipeline
run on ten models is worse than no run.

Decisions
- Add Qwen/Qwen2.5-Math-1.5B (baseline for the DeepSeek distill).
- Add gemma-4-E2B-it as the LAST model, optional. If it fails to load or runs
  out of memory, record that and move on. Skip gemma-4-E4B-it.
- Disk: process one model at a time. Download, compute, verify the output file,
  then delete only that model's weights. Never delete anything in my Hugging Face
  cache that this project did not download.
- Thinking modes off for Qwen3 and SmolLM3. For the DeepSeek distill, prefill an
  empty thinking block so the next token is the answer, and label its results
  as using that workaround.
- Use one fixed date for any chat template that injects a date, and one precision
  for all models. Record both in the results.

Gate A: rerun scripts/check_access.py. Drop any model I still lack access to and
list it in the final report.

Gate B: run the smoke test. If half precision gives NaNs on MPS, switch every
model to float32.

Gate C (validation): generate log-probabilities locally for google/gemma-2-2b-it
and meta-llama/Llama-3.2-3B-Instruct on all 54 prompts, following the authors'
prompt setup exactly. Compare against the authors' precomputed files: Spearman
correlation over the Big Two nouns for each prompt, plus top-15 token overlap on
the "greatest thing ever" prompt. If the median correlation for either model is
below 0.9, stop and report what differs. Do not adjust things until it passes
without telling me what you changed.

Survey (only if all gates pass): for each remaining model, smallest first,
compute next-token log-probabilities for all 54 prompts and save them in the
same format as the authors' files so their analysis code can read them. Run the
long job under caffeinate so the Mac does not sleep. Keep a run log with time,
disk free, and status per model.

Analysis
- Per model: median rank of agency nouns and of communion nouns, separately for
  positive and negative prompts, using Big Two nouns that are single tokens in
  that model's vocabulary. Report how many words each model's score is based on.
- Repeat on only the words shared by every model, and report how many those are.
- One chart: agency minus communion gap per model on positive prompts, with the
  paper's 12 models included as reference points.
- One table per model: top 15 and bottom 15 tokens for the "greatest thing
  ever" prompt.

Write results.md with what ran, what failed, the chart and tables, and every
caveat (thinking workaround, multimodal models, tokenizer differences, date,
precision). Report the data and flag surprises. Do not interpret beyond that.
