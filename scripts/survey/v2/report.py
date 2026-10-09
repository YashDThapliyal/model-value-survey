"""Write results_v2.md and its charts from results_v2/tables.pkl.

Usage: .venv/bin/python scripts/survey/v2/report.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from analysis_report import (AXIS, FAMILY_COLORS, GRID, INK, INK_2, OTHER_COLOR,  # noqa: E402
                             SURFACE, family_group)
from extract import LEAD_IN_WORDS, MWLR_TARGETS, TOP_PROMPT  # noqa: E402
from text_v2 import CAVEATS, METHOD  # noqa: E402

ROOT = HERE.parents[2]
OUT = ROOT / "results_v2"
TIE_FLAG = 0.5
REF_SHORT = {"google/gemma-2-2b-it": "Gemma-2-2B-IT", "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B-Instruct"}


def label(r: dict) -> str:
    return f"{r['label']} ({'ours' if r['source'] == 'local' else 'paper'})"


def flags(r: dict) -> str:
    out = []
    if r["flag_lead_in"]:
        out.append(f"LEAD-IN {r['lead_in_pos']:.2f}")
    if r["flag_low_mass"]:
        out.append("LOW BIG-TWO MASS")
    return ", ".join(out) or "—"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def section_flags(rows: list[dict], meta: dict) -> str:
    bad = [r for r in rows if r["flag_lead_in"] or r["flag_low_mass"]]
    body = table(["Model", "Flag", "Lead-in mass (pos)", "Big Two variant mass (pos)", f"Top token on {TOP_PROMPT}"],
                 [[label(r), flags(r), f"{r['lead_in_pos']:.3f}", f"{r['big2_pos']:.3f}",
                   f"`{r['top_token']!r}` p={r['top_prob']:.2f}"] for r in bad])
    tie_rows = [[label(r), f"{r['ties_own']['tied_share']:.0%}", f"{r['ties_shared']['tied_share']:.0%}"]
                for r in rows if max(r["ties_own"]["tied_share"], r["ties_shared"]["tied_share"]) > TIE_FLAG]
    ties = ("\n**Rank-based measures dominated by ties.** This rule was set *after* the tie table in section 5 was "
            f"computed, so it is not pre-declared: more than {TIE_FLAG:.0%} of ranked words tied on an average prompt. "
            "For these models the rank gap (v1 chart and section 2) rests mostly on average-rank tie-breaking. "
            "The mass measures are unaffected.\n\n"
            + table(["Model", "Tied share (own words)", "Tied share (shared words)"], tie_rows) + "\n")
    neg_mass = [r["mass_neg"]["agency"].value + r["mass_neg"]["communion"].value for r in rows]
    n_low_neg = sum(m < meta["low_mass_flag"] for m in neg_mass)
    ties += (f"\n**Negative prompts carry little Big Two mass.** Applying the same pre-declared low-mass rule to the 27 "
             f"negative prompts flags {n_low_neg} of {len(rows)} models (largest negative-prompt Big Two mass: "
             f"{max(neg_mass):.4f}). Every negative-prompt mass value, and therefore the valence-controlled mass gap "
             f"in section 2, rests on less than {meta['low_mass_flag']:.0%} of the probability for those models.\n")
    return (f"Pre-declared rules: **lead-in dominated** if the mean lead-in mass over the 27 positive prompts is "
            f"> {meta['lead_in_flag']}; **low Big Two mass** if the mean summed variant probability on the shared "
            f"Big Two words over the positive prompts is < {meta['low_mass_flag']}. Lead-in-dominated models are "
            f"excluded from both charts. Their rows stay in the tables, marked.\n\n{body}\n{ties}")


def section_mass(rows: list[dict], meta: dict) -> str:
    n = meta["n_mass_shared"]
    hdr = ["Model", "Flag", "Agency pos", "Communion pos", "A − C pos", "Agency neg", "Communion neg", "A − C neg"]
    body = table(hdr, [[label(r), flags(r)] + [r[k][m].fmt(4) for k in ("mass_pos", "mass_neg")
                                               for m in ("agency", "communion", "a_minus_c")] for r in rows])
    own_hdr = ["Model", "Words with ≥1 variant (A/C)", "Variant tokens", "Agency pos", "Communion pos", "A − C pos"]
    own = table(own_hdr, [[label(r), f"{r['n_words_variant_own']['Agency']}/{r['n_words_variant_own']['Communion']}",
                           r["n_variant_tokens_own"]] + [r["mass_own_pos"][m].fmt(4) for m in ("agency", "communion", "a_minus_c")]
                          for r in rows])
    return (f"Total probability (summed over variants, then over words) on Agency vs Communion words. Each value is "
            f"the mean over the valence's 27 prompts, with a bootstrap 95% interval in brackets.\n\n"
            f"**Shared word set:** {n['Agency']} Agency + {n['Communion']} Communion words that have at least one "
            f"single-token variant in all 25 models.\n\n{body}\n\n![mass chart](results_v2/mass_positive.png)\n\n"
            f"**Each model's own covered words** (counts differ by tokenizer, so not comparable across models):\n\n{own}\n")


def section_valence(rows: list[dict], meta: dict) -> str:
    n = meta["n_exact_shared"]
    body = table(["Model", "Flag", "Rank gap pos", "Rank gap neg", "Rank gap pos − neg", "Mass (A − C) pos − neg"],
                 [[label(r), flags(r), r["rank_pos"].fmt(2), r["rank_neg"].fmt(2), r["rank_valence_gap"].fmt(2),
                   r["mass_valence_gap"].fmt(4)] for r in rows])
    return (f"**Rank gap:** the existing measure, median rank(Agency) − median rank(Communion), on the {n['Agency']}+{n['Communion']} "
            f"lowercase single-token words shared by all models (authors' method). The point values equal v1 "
            f"(`results/per_model_shared_vocab.csv`). Negative means Agency ranked higher.\n\n"
            f"**Mass:** mean (Agency − Communion) variant-merged probability on the shared variant set. Negative means more mass on Communion.\n\n"
            f"Positive minus negative removes any shift common to both valences. Prompts were resampled independently within each valence.\n\n"
            f"{body}\n\n![valence chart](results_v2/valence_controlled.png)\n\n"
            "This chart is nearly identical to the positive-prompt mass chart in section 1, because negative-prompt "
            "Big Two mass is close to zero for every model (see the flags section).\n")


def section_mwlr(rows: list[dict]) -> str:
    parts = ["MWLR(token) = ½(p_model + p_ref)·(log p_model − log p_ref). Positive means the model puts more probability "
             "on the word than the reference. The references are our local runs of the two Gate C models.\n",
             f"- **merged:** p = summed probability over the word's variants, on `{TOP_PROMPT}` (the Figure 3 prompt).\n"
             f"- **authors' rule:** the vendor `generate_figure_3.py` logic applied pairwise. Each model keeps its maximum log-prob per decoded "
             f"string, decoded strings shared by model and reference are kept, and the result is the mean of per-token MWLR over strings with "
             f"`strip().lower()` equal to the word, on `{TOP_PROMPT}`.\n"
             f"- **merged, 27 positive prompts:** the merged value averaged over the positive prompts, with a bootstrap 95% interval.\n"]
    for ref, short in REF_SHORT.items():
        hdr = ["Model", "Flag"] + [f"{w} {kind}" for w in MWLR_TARGETS for kind in ("merged", "authors' rule", "merged, pos mean [95% CI]")]
        body = [[label(r), flags(r)] + [v for w in MWLR_TARGETS for v in (
            f"{r['mwlr'][(ref, w)]['merged_top']:.4f}", f"{r['mwlr'][(ref, w)]['authors_top']:.4f}",
            r["mwlr"][(ref, w)]["merged_pos_mean"].fmt(4))] for r in rows]
        parts.append(f"\n**Reference: {short}**\n\n" + table(hdr, body) + "\n")
    return "\n".join(parts)


def section_answer(rows: list[dict]) -> str:
    classes = ["lead-in", "big2-variant", "other-word", "fragment", "other"]
    body = table(["Model", "Flag", f"Top token ({TOP_PROMPT})", "p", "Class"] + [f"{c} (pos mean)" for c in classes],
                 [[label(r), flags(r), f"`{r['top_token']!r}`", f"{r['top_prob']:.3f}", r["top_class"]]
                  + [f"{r['class_shares_pos'].get(c, 0.0):.3f}" for c in classes] for r in rows])
    words = ", ".join(f"`{w}`" for w in sorted(LEAD_IN_WORDS))
    return ("Every vocabulary token is classified once, using its decoded string with whitespace stripped and "
            "matched case-insensitively. Class masses are probability summed over the whole vocabulary, then "
            "averaged over the 27 positive prompts.\n\n"
            f"- **lead-in:** in the declared set ({words}), or made only of punctuation or whitespace (`**`, `\"`, `\\n\\n`).\n"
            "- **big2-variant:** a case or leading-space variant of a Big Two noun.\n"
            "- **other-word:** alphabetic, at least 3 letters, and in `/usr/share/dict/words`.\n"
            "- **fragment:** any other alphabetic token (`Un`, `Im`, single letters). Note: the macOS dictionary contains some fragments, e.g. `creat`.\n"
            "- **other:** digits, emoji and mixed tokens.\n\n" + body + "\n")


def section_ties(rows: list[dict], meta: dict) -> str:
    def cells(t: dict) -> list[str]:
        return [t["n_ranked"], f"{t['tied_mean']:.1f}", t["tied_max"], f"{t['tied_share']:.0%}",
                f"{t['distinct_mean']:.1f}", f"{t['prompts_with_ties']}/54"]
    hdr = ["Model", "Ranked words", "Tied words / prompt (mean)", "max", "share", "distinct log-probs (mean)", "prompts with ties"]
    own = table(hdr, [[label(r)] + cells(r["ties_own"]) for r in rows])
    shr = table(hdr, [[label(r)] + cells(r["ties_shared"]) for r in rows])
    return ("A word counts as tied when at least one other ranked Big Two word has exactly the same log-probability on "
            "that prompt.\n\n"
            "**How the current code breaks ties:** `rank(method='average', ascending=False)`, the same as the authors. Tied words "
            "all receive the mean of the ranks they span, and the per-category median is taken over those ranks. This "
            "produces .5 medians and lets ties span Agency and Communion words alike. Ties are consistent with the "
            "bfloat16 logits, which have only 8 bits of mantissa, so log-probs fall on a coarse grid. For example, "
            "Llama-3.2-3B-Instruct's 128,256 log-probs on `greatest_ever_one` take only 2,770 distinct values, and "
            "near the top they are spaced in multiples of 1/16.\n\n"
            f"**Each model's own ranked set:**\n\n{own}\n\n**Shared ranked set ({sum(meta['n_exact_shared'].values())} words):**\n\n{shr}\n")


def dot_chart(rows: list[dict], get, title: str, xlabel: str, path: Path) -> None:
    rows = sorted([r for r in rows if not r["flag_lead_in"]], key=lambda r: get(r).value)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.33 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    for y, r in enumerate(rows):
        e = get(r)
        color = FAMILY_COLORS.get(family_group(r["family"]), OTHER_COLOR)
        ax.hlines(y, e.lo, e.hi, color=color, linewidth=2, zorder=2)
        ax.scatter(e.value, y, s=56, zorder=3, edgecolor=color, linewidth=2,
                   facecolor=color if r["source"] == "local" else SURFACE)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], color=INK, fontsize=8.5)
    ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    data = pickle.loads((OUT / "tables.pkl").read_bytes())
    rows, meta = data["rows"], data["meta"]
    note = ("Filled = our runs, hollow = paper's files. Bars = bootstrap 95% interval over prompts. "
            "Lead-in-dominated models excluded.")
    dot_chart(rows, lambda r: r["mass_pos"]["a_minus_c"],
              "Agency − Communion probability mass, positive prompts",
              f"mean over 27 positive prompts of P(Agency words) − P(Communion words), variants merged\n"
              f"← more mass on Communion · more mass on Agency →\n{note}", OUT / "mass_positive.png")
    dot_chart(rows, lambda r: r["mass_valence_gap"],
              "Valence-controlled mass gap (positive − negative prompts)",
              f"[A − C mass](positive) − [A − C mass](negative)\n{note}", OUT / "valence_controlled.png")
    excluded = [label(r) for r in rows if r["flag_lead_in"]]
    doc = "\n".join([
        "# Results v2: probability-mass measures (analysis only)", "",
        METHOD, "",
        "## Models whose results are not interpretable", "", section_flags(rows, meta),
        "Excluded from charts: " + "; ".join(excluded) + ".", "",
        "## 1. Variant-merged word probability", "", section_mass(rows, meta),
        "## 2. Valence-controlled gap", "", section_valence(rows, meta),
        "## 3. Freedom vs Love (MWLR, as in the paper's Figure 3)", "", section_mwlr(rows),
        "## 4. Answer rate", "", section_answer(rows),
        "## 5. Ties", "", section_ties(rows, meta),
        "## Caveats", "", CAVEATS, "",
    ])
    (ROOT / "results_v2.md").write_text(doc)
    print("wrote results_v2.md")


if __name__ == "__main__":
    main()
