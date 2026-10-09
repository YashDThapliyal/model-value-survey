"""Append the results section to results_v3.md (pre-registration above it is never rewritten)."""

from __future__ import annotations

import csv
from pathlib import Path

import metrics as V

ROOT = Path(__file__).resolve().parents[3]
MARK = "# Results\n"


def t(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def ordered(models: list[str], short: dict[str, str]) -> list[str]:
    return sorted(models, key=lambda m: list(short).index(m) if m in short else 99)


def predictions(p1, p2, short) -> str:
    ok, gmin, ngmax, gk, nk = p1
    lines = [f"- **P1: {'HELD' if ok else 'FAILED'}.** The lowest Gemma communion answer share is {gmin:.3f} "
             f"({short.get(gk, gk)}). The highest non-Gemma share is {ngmax:.3f} ({short.get(nk, nk)})."]
    rows = [[V.NEW_QUESTIONS[q] + " *(new)*", f"{r['gemma_mean']:.3f}", f"{r['llama_mean']:.3f}",
             "yes" if r["mean_rule"] else "no", "yes" if r["strict"] else "no"]
            for q, r in p2["per_question"].items()]
    lines.append(f"- **P2: {'HELD' if p2['holds'] else 'FAILED'}.** The Gemma mean was above the Llama mean on "
                 f"{p2['n_questions_gemma_higher']} of 4 new questions (at least 3 needed).\n")
    lines.append(t(["New question", "Gemma mean communion share", "Llama mean", "Gemma > Llama (decides P2)",
                    "Every Gemma > every Llama"], rows))
    return "\n".join(lines)


def status(models, short) -> str:
    log = list(csv.DictReader((ROOT / "runs" / "v3" / "run_log.csv").open()))
    last = {r["model"]: r for r in log}
    plan = ordered(list(short), short)
    rows = [[short.get(m, m), m, last[m]["status"] if m in last else "not run",
             last[m]["note"].replace(" | ", "; ")[:160] if m in last else ""] for m in plan]
    return t(["Model", "ID", "Status", "Note"], rows)


def part_a(A, A_nouns, models, short) -> str:
    rows = [[short[m]] + [A[m]["est"][c].fmt(3) for c in V.CLASSES]
            + [", ".join(f"{w} {g:.0%}" for w, g in A[m]["greedy_cls"].items() if g)] for m in models]
    words = [[short[m], ", ".join(f"{w} ({s:.0%})" for w, _, s in A[m]["top_words"])] for m in models]
    return ("Shares are means over 27 prompts of each prompt's share among its 20 samples; 95% bootstrap intervals over prompts.\n\n"
            + t(["Model", "Communion", "Agency", "Other", "Unextractable", "Greedy answers (class share)"], rows)
            + "\n\n![answer shares](results_v3/answer_shares_original.png)\n\n**Top answer words (10 most common sampled words, share of all samples):**\n\n"
            + t(["Model", "Top words"], words) + "\n")


def part_b(B, models, short) -> str:
    out = ["These four questions are **new and not from the paper**. Each was asked in 3 phrasings, with 20 samples per "
           "phrasing. Intervals resample the 60 samples."]
    for q, text in V.NEW_QUESTIONS.items():
        rows = [[short[m]] + [B[m][q][c].fmt(3) for c in V.CLASSES]
                + [", ".join(f"{w} {n}" for w, n in B[m][q]["top_words"].items())] for m in models]
        out += [f"\n### \"{text}\" (new question, not from the paper)\n",
                t(["Model", "Communion", "Agency", "Other", "Unextractable", "Top 5 words (count of 60)"], rows),
                f"\n![{q}](results_v3/answer_shares_{q}.png)\n"]
    rows = [[short[m]] + [B[m]["pooled"][c].fmt(3) for c in V.CLASSES] for m in models]
    out += ["\n### Pooled over the 4 new questions (new, not from the paper)\n",
            t(["Model", "Communion", "Agency", "Other", "Unextractable"], rows)]
    return "\n".join(out) + "\n"


def part_c(A, A_nouns, ft, models, short) -> str:
    rows = []
    for m in models:
        if m not in ft:
            rows.append([short[m], "n/a (no first-token CSV)"] + [""] * 6)
            continue
        sc, sa = A[m]["est"]["communion"].value, A[m]["est"]["agency"].value
        mc, ma = ft[m]["mass_c"].value, ft[m]["mass_a"].value
        flags = V.part_c_flag(sc, sa, mc, ma)
        rows.append([short[m], f"{mc:.3f}", f"{ma:.3f}", f"{sc:.3f}", f"{sa:.3f}",
                     f"{A_nouns[m]['est']['communion'].value:.3f}", f"{A_nouns[m]['est']['agency'].value:.3f}",
                     "**FLAG**: " + "; ".join(flags) if flags else "agree"])
    return t(["Model", "First-token communion mass", "First-token agency mass", "Sampled communion share",
              "Sampled agency share", "Sampled communion (263-noun dict)", "Sampled agency (263-noun dict)",
              "Part C"], rows) + "\n"


def part_e(A, ft, models, short) -> str:
    rows = [[short[m], ft[m]["comm_share_of_big2"].fmt(3), ft[m]["top1"].fmt(3), ft[m]["entropy"].fmt(2),
             f"{A[m]['modal_share']:.3f} (`{A[m]['modal_word']}`)"] if m in ft else [short[m], "n/a", "n/a", "n/a",
             f"{A[m]['modal_share']:.3f}"] for m in models]
    return t(["Model", "(a) Communion share of Big Two first-token probability", "(b) Top-1 first-token probability",
              "(b) First-token entropy (nats)", "(b) Modal sampled answer share"], rows) + "\n"


def write_results(path: Path, A, A_nouns, B, ft, p1, p2, n_shared, models, short) -> None:
    models = ordered(models, short)
    text = path.read_text()
    pre = text.split(MARK)[0]
    part_d = (ROOT / "results_v3" / "part_d.md")
    body = "\n".join([
        MARK, "## Predictions: verdicts", "", predictions(p1, p2, short), "",
        "## Status of every model in the plan", "", status(models, short), "",
        "## Gate", "", (ROOT / "runs" / "v3" / "gate.md").read_text().split("\n", 1)[1], "",
        "## Part D: authors' Figure 3, unmodified", "",
        part_d.read_text().split("\n", 1)[1].replace("\n## ", "\n### ") if part_d.exists() else "_Part D output missing._", "",
        "## Part A: sampled answers on the original 27 positive prompts (Q1)", "", part_a(A, A_nouns, models, short),
        "## Part B: new questions (Q2)", "", part_b(B, models, short),
        "## Part C: first-token mass vs sampled answers", "",
        f"First-token mass uses the results_v2 section 1 method on the {n_shared['Agency']} Agency + {n_shared['Communion']} "
        "Communion nouns with single-token variants in every local and paper model. The flag rule is pre-registered above.\n",
        part_c(A, A_nouns, ft, models, short),
        "## Part E: concentration (Q3)", "", part_e(A, ft, models, short),
        "## Caveats", "", (ROOT / "scripts" / "survey" / "v3" / "caveats_v3.md").read_text(),
    ])
    path.write_text(pre + body)
