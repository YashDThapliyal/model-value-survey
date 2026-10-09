"""Blog-readable charts for results_v3: models grouped by family, Gemma generations in order."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_report import AXIS, GRID, INK, INK_2, SURFACE  # noqa: E402

COMMUNION_COLOR, AGENCY_COLOR = "#eb6834", "#2a78d6"
# Display order: family blocks, Gemma generations oldest -> newest, then size.
ORDER = [
    ("Gemma", ["google/gemma-1.1-2b-it", "google/gemma-2-2b-it", "google/gemma-3-270m-it",
               "google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-4-E2B-it"]),
    ("Llama", ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]),
    ("Qwen", ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]),
    ("DeepSeek", ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]),
    ("Phi", ["microsoft/Phi-4-mini-instruct"]),
    ("SmolLM", ["HuggingFaceTB/SmolLM3-3B"]),
]
SHORT = {
    "google/gemma-1.1-2b-it": "Gemma 1.1 2B", "google/gemma-2-2b-it": "Gemma 2 2B",
    "google/gemma-3-270m-it": "Gemma 3 270M", "google/gemma-3-1b-it": "Gemma 3 1B",
    "google/gemma-3-4b-it": "Gemma 3 4B", "google/gemma-4-E2B-it": "Gemma 4 E2B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B", "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen 2.5 3B", "Qwen/Qwen2.5-Math-1.5B": "Qwen 2.5 Math 1.5B (base)",
    "Qwen/Qwen3-1.7B": "Qwen 3 1.7B", "Qwen/Qwen3-4B": "Qwen 3 4B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek R1 Distill 1.5B (thinks)",
    "microsoft/Phi-4-mini-instruct": "Phi 4 mini", "HuggingFaceTB/SmolLM3-3B": "SmolLM 3 3B",
}


def layout(available: set[str]) -> list[tuple[str, list[str]]]:
    return [(fam, [m for m in ms if m in available]) for fam, ms in ORDER if any(m in available for m in ms)]


def share_chart(est: dict[str, dict], title: str, subtitle: str, path: Path) -> None:
    """est[model] = {'communion': Est, 'agency': Est}. Paired horizontal bars with 95% CIs."""
    groups = layout(set(est))
    rows: list[str | tuple[str]] = []
    for fam, models in groups:
        rows.append((fam,))
        rows.extend(models)
    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.42 * len(rows)))
    fig.patch.set_facecolor(SURFACE)
    labels, h = [], 0.36
    for y, row in enumerate(rows):
        if isinstance(row, tuple):
            labels.append(row[0].upper())
            continue
        labels.append(SHORT.get(row, row))
        for off, key, color in ((-h / 2, "communion", COMMUNION_COLOR), (h / 2, "agency", AGENCY_COLOR)):
            e = est[row][key]
            ax.barh(y + off, e.value, height=h, color=color, zorder=2)
            ax.errorbar(e.value, y + off, xerr=[[e.value - e.lo], [e.hi - e.value]], fmt="none",
                        ecolor=INK_2, elinewidth=1, capsize=2, zorder=3)
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if isinstance(row, tuple) else "normal")
        tick.set_color(INK if isinstance(row, tuple) else INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of sampled answers (20 samples per prompt, temperature 1.0); bars = 95% bootstrap interval",
                  color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, colors=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COMMUNION_COLOR, AGENCY_COLOR)]
    ax.legend(handles, ["Communion answer (love, family, friendship…)", "Agency answer (freedom, success, achievement…)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=9.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
