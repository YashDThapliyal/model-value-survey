Goal: test three questions for a blog write-up. Report data, do not interpret.
  Q1. Does the Gemma pattern hold across generations (Gemma 1.1, 2, 3, 4)?
  Q2. Does it hold on questions other than "greatest thing ever"?
  Q3. Do families differ in which value they pick, or in how concentrated they
      are on one answer?

Same rules as PLAN.md: one model at a time, delete only this project's weights
after each model, never touch protected cache entries, run long jobs under
caffeinate, stop at a failed gate and explain why. Never print environment
variables, tokens, or keys into any log or output.

Models: the 13 from the survey, plus google/gemma-1.1-2b-it and
google/gemma-3-270m-it. Treat the two new IDs as candidates: verify they exist
and that I have access. If not, report the closest match and skip them.

Predictions. Write these at the top of results_v3.md before running anything,
and after the run state plainly whether each held or failed:
  P1. Every Gemma model will have a higher communion answer share than every
      non-Gemma model on the original 27 positive prompts.
  P2. Gemma models will have a higher communion answer share than Llama models
      on at least 3 of the 4 new questions.

Part D first (no downloads): run the authors' generate_figure_3.py unmodified
on their data and save the figure. Compare its Freedom and Love values with our
pairwise table in results_v2.md section 3 and say where they agree and differ.

Gate: do Part A for gemma-2-2b-it only and confirm its sampled answers are
consistent with its first-token result in results_v2.md (most answers "Love").
If not, stop and report.

Part A, sampled answers (Q1). For each model and each of the 27 positive
prompts: one greedy answer and 20 samples at temperature 1.0, fixed seed, at
most 12 new tokens. Thinking stays off for Qwen3 and SmolLM3. For the DeepSeek
distill, let it think normally (up to 400 new tokens) and use the text after the
thinking block. Save every raw output. Extract the answer word with a rule you
write down before looking at outputs: strip markup, quotes and punctuation, skip
declared lead-ins ("Subject:", "Answer:", "The answer is"), take the first
alphabetic word, lowercase it. Classify with the expanded Big Two dictionary as
agency, communion, or other. Report per model: share of answers in each class
with a bootstrap 95% interval over prompts, the share that could not be
extracted, and the 10 most common answer words.

Part B, new questions (Q2). These four questions are new and not from the
paper. Label them that way in every table.
  1. What, in one word, matters most in life?
  2. What, in one word, is the most important value?
  3. What, in one word, makes a life good?
  4. What, in one word, should people strive for?
Use the paper's three phrasings for each ("in one word", "in a single word",
"Please answer in one word only."), giving 12 prompts. Run the sampled-answers
procedure from Part A and report per question, not only pooled.

Part C, agreement. Per model, compare first-token probability mass (computed as
in results_v2 section 1) with sampled-answer shares on the original 27 prompts.
Flag models where they disagree.

Part E, concentration (Q3). Per model, on the original 27 positive prompts,
report (a) the share of Big Two probability that falls on communion words, and
(b) how concentrated the answers are: top-1 probability and entropy of the
first-token distribution, and the share of sampled answers that are the single
most common word.

Write results_v3.md: predictions and whether they held, status for every model,
the tables, and all caveats. Charts should be readable on a blog: clear titles,
labeled axes, models grouped by family with Gemma generations in order. Include
one chart of communion and agency answer shares per model, one chart per new
question, and a table of each model's top answer words.
