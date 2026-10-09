"""Chart and markdown output helpers for analyze.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Reference palette (dataviz skill): the first three categorical slots are the only
# ones that validate all-pairs for a dot chart, so families beyond three fold into a
# neutral "other". Every row is also labelled by name, so colour is never alone.
FAMILY_COLORS = {"llama": "#2a78d6", "gemma": "#eb6834", "qwen": "#1baf7a"}
OTHER_COLOR = "#898781"
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SOURCE_TITLES = {"local": "Our runs (local)", "paper": "Paper reference (Christian et al. data)"}


def family_group(family: str) -> str:
    fam = family.lower()
    return next((k for k in FAMILY_COLORS if fam.startswith(k)), "other")


def _panel(ax: plt.Axes, rows: pd.DataFrame, title: str, filled: bool) -> None:
    rows = rows.sort_values("gap_median", ascending=False).reset_index(drop=True)
    ys = range(len(rows))
    colors = [FAMILY_COLORS.get(family_group(f), OTHER_COLOR) for f in rows.family]
    ax.hlines(list(ys), 0, rows.gap_median, color=colors, linewidth=2, zorder=2)
    ax.scatter(rows.gap_median, list(ys), s=64, zorder=3,
               facecolor=colors if filled else SURFACE, edgecolor=colors, linewidth=2)
    ax.set_yticks(list(ys), rows.label, color=INK, fontsize=9)
    ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", colors=INK_2, length=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)


def _legend(fig: plt.Figure, families: list[str]) -> None:
    groups = sorted({family_group(f) for f in families},
                    key=lambda g: list(FAMILY_COLORS).index(g) if g in FAMILY_COLORS else 99)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8,
                      color=FAMILY_COLORS.get(g, OTHER_COLOR), label=g.capitalize())
               for g in groups]
    handles += [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=INK_2, label="local (filled)"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE,
               markeredgecolor=INK_2, markeredgewidth=2, label="paper (hollow)"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.005, 0.97))


def plot_gap(positive: pd.DataFrame, n_shared: dict[str, int], out_stem: Path) -> list[Path]:
    """Horizontal dot chart of the positive-prompt agency-minus-communion gap."""
    groups = [(s, positive[positive.source == s]) for s in ("local", "paper")]
    groups = [(s, g) for s, g in groups if len(g)]
    heights = [len(g) for _, g in groups]
    fig, axes = plt.subplots(len(groups), 1, sharex=True, squeeze=False,
                             figsize=(8.5, 1.3 + 0.32 * sum(heights)),
                             gridspec_kw={"height_ratios": heights})
    fig.patch.set_facecolor(SURFACE)
    for ax, (source, rows) in zip(axes[:, 0], groups):
        _panel(ax, rows, SOURCE_TITLES[source], filled=source == "local")
    n_text = ", ".join(f"{k} {v}" for k, v in n_shared.items())
    axes[-1, 0].set_xlabel(
        "Gap = median rank(Agency) − median rank(Communion), positive prompts\n"
        "← negative: Agency nouns ranked higher · positive: Communion nouns ranked higher →\n"
        f"Shared single-token Big Two nouns across all models: {n_text}",
        color=INK_2, fontsize=8.5)
    _legend(fig, list(positive.family))
    fig.suptitle("Agency vs Communion on positive prompts", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = [out_stem.with_suffix(ext) for ext in (".png", ".svg")]
    for path in paths:
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return paths


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "\\n")


def _md_rows(df: pd.DataFrame) -> list[str]:
    return [f"| {r.rank} | {r.token_id} | `` {_md_cell(repr(r.token_decoded))} `` | {r.logprob:.4f} |"
            for r in df.itertuples()]


def top_bottom_markdown(label: str, model_id: str, prompt: str, n_vocab: int,
                        top: pd.DataFrame, bottom: pd.DataFrame) -> str:
    header = ["| rank | token_id | token_decoded (repr) | log-prob |", "|---:|---:|---|---:|"]
    return "\n".join([
        f"## {label}", "", f"`{model_id}` · prompt `{prompt}` · vocab rows {n_vocab}", "",
        f"**Top {len(top)}**", "", *header, *_md_rows(top), "",
        f"**Bottom {len(bottom)}**", "", *header, *_md_rows(bottom), "",
    ])
