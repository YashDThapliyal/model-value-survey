"""Data loading and Big Two ranking helpers for analyze.py.

Ranking replicates vendor figures/generate_figure_2.py:
  * a Big Two noun is "in" a model if it is `isin` that model's token_decoded column;
  * per prompt, inner-merge the noun list with the model's log-probs on token_decoded
    (duplicate token rows are kept, exactly like the authors' merge), then rank the
    merged rows' log-probs descending with method='average' (rank 1 = most likely);
  * median rank per (prompt, Category) is the unit of analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = REPO_ROOT / "vendor" / "reward_models_inherit_value_biases_from_pretraining"
VENDOR_LOGPROBS = VENDOR_ROOT / "data" / "logprobs"
BIG2_PATH = VENDOR_ROOT / "data" / "corpora" / "dict_big2_nouns.csv"
CATEGORIES = ("Agency", "Communion")
ID_COLS = ["token_id", "token_name", "token_decoded"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    source: str  # "local" | "paper"
    path: Path
    family: str
    role: str
    prompt_mode: str
    multimodal: bool = False
    display_name: str = ""
    meta: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def key(self) -> str:
        return f"{self.source}:{self.model_id}"

    @property
    def label(self) -> str:
        base = self.display_name or self.model_id.split("/")[-1]
        flags = []
        if self.prompt_mode == "chat_empty_think_prefill":
            flags.append("empty-think")
        if self.source == "paper" and self.role == "pretrained":
            flags.append("pretrained")
        elif self.prompt_mode == "raw" or self.role == "base":
            flags.append("base/raw")
        if self.multimodal:
            flags.append("multimodal")
        return f"{base} [{', '.join(flags)}]" if flags else base


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def is_good(key: str) -> bool:
    """Authors' valence rule: positive prompt iff key contains greatest/best/good."""
    return "greatest" in key or "best" in key or "good" in key


def load_prompt_keys() -> list[str]:
    with open(VENDOR_ROOT / "config" / "prompts.yaml") as f:
        return list(yaml.safe_load(f).keys())


def load_big2() -> pd.DataFrame:
    return pd.read_csv(BIG2_PATH)


def paper_specs(logprob_dir: Path = VENDOR_LOGPROBS) -> list[ModelSpec]:
    with open(VENDOR_ROOT / "config" / "models.yaml") as f:
        models = yaml.safe_load(f)
    return [
        ModelSpec(
            model_id=m["name"], source="paper",
            path=logprob_dir / f"{safe_name(m['name'])}.csv",
            family=m.get("family", ""), role=m.get("type", ""),
            prompt_mode="raw" if m.get("type") == "pretrained" else "chat",
            display_name=m.get("display_name", ""),
        )
        for m in models
    ]


def _read_meta(csv_path: Path) -> dict | None:
    meta_path = csv_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def local_specs(config_path: Path, local_dir: Path,
                require_meta: bool = True) -> tuple[list[ModelSpec], list[str]]:
    """Return (available local specs, notes about skipped models)."""
    with open(config_path) as f:
        models = yaml.safe_load(f)["models"]
    specs: list[ModelSpec] = []
    notes: list[str] = []
    for m in models:
        if m.get("status") == "skipped":
            continue
        path = local_dir / f"{safe_name(m['id'])}.csv"
        meta = _read_meta(path)
        if not path.exists():
            notes.append(f"{m['id']}: missing {path.name}")
            continue
        if meta is None and require_meta:
            notes.append(f"{m['id']}: no .meta.json yet (generation incomplete?)")
            continue
        specs.append(ModelSpec(
            model_id=m["id"], source="local", path=path,
            family=m.get("family", ""), role=m.get("role", ""),
            prompt_mode=m.get("prompt_mode", "chat"),
            multimodal=bool(m.get("multimodal", False)),
            meta=meta or {},
        ))
    return specs, notes


NUMERIC_NA = ["", "nan", "NaN"]


def read_vocab(path: Path) -> pd.Series:
    """token_decoded column only; literal strings like 'nan'/'None' stay tokens.

    The authors use default read_csv, which turns tokens such as 'null', 'None',
    'nan', 'NA' into NaN. No Big Two noun is such a string, so Big Two results are
    identical either way; it only matters for the full-vocab top/bottom tables.
    """
    return pd.read_csv(path, usecols=["token_decoded"], keep_default_na=False)["token_decoded"]


def read_logprobs(path: Path, columns: list[str]) -> pd.DataFrame:
    """token_id, token_decoded and the requested prompt columns (NA only in numerics)."""
    return pd.read_csv(path, usecols=["token_id", "token_decoded", *columns],
                       keep_default_na=False,
                       na_values={c: NUMERIC_NA for c in columns})


def words_in_vocab(big2: pd.DataFrame, vocab: pd.Series) -> pd.DataFrame:
    return big2[big2.token_decoded.isin(vocab)]


def per_prompt_medians(words: pd.DataFrame, lp: pd.DataFrame,
                       prompts: list[str]) -> pd.DataFrame:
    """Median rank per (prompt, Category), authors' merge-then-rank."""
    merged = pd.merge(words, lp[["token_decoded", *prompts]], on="token_decoded", how="inner")
    frames = [
        merged.assign(rank=merged[key].rank(method="average", ascending=False))
        .groupby("Category", as_index=False)["rank"].median()
        .assign(prompt=key, valence="positive" if is_good(key) else "negative")
        for key in prompts
    ]
    return pd.concat(frames, ignore_index=True)


def duplicate_rows(words: pd.DataFrame, lp: pd.DataFrame) -> pd.DataFrame:
    """Words that match more than one token row (kept as duplicates by the merge)."""
    counts = lp.loc[lp.token_decoded.isin(words.token_decoded), "token_decoded"].value_counts()
    dups = counts[counts > 1]
    return pd.DataFrame({"token_decoded": dups.index, "n_rows": dups.values})


def summarise_medians(medians: pd.DataFrame) -> pd.DataFrame:
    """Per valence: median and mean across prompts of per-prompt median ranks (wide)."""
    agg = medians.groupby(["valence", "Category"])["rank"].agg(["median", "mean", "count"])
    wide = agg.unstack("Category")
    out = pd.DataFrame({
        "valence": wide.index,
        "agency_median_rank": wide[("median", "Agency")].values,
        "communion_median_rank": wide[("median", "Communion")].values,
        "agency_mean_rank": wide[("mean", "Agency")].values,
        "communion_mean_rank": wide[("mean", "Communion")].values,
        "n_prompts": wide[("count", "Agency")].values,
    })
    return out.assign(
        gap_median=out.agency_median_rank - out.communion_median_rank,
        gap_mean=out.agency_mean_rank - out.communion_mean_rank,
    )


def model_result(spec: ModelSpec, words: pd.DataFrame, lp: pd.DataFrame,
                 prompts: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One model's summary rows (one per valence) plus its duplicate table."""
    used = words[words.token_decoded.isin(lp.token_decoded)]
    dups = duplicate_rows(used, lp)
    summary = summarise_medians(per_prompt_medians(used, lp, prompts))
    meta_cols = {
        "source": spec.source, "model_id": spec.model_id, "label": spec.label,
        "family": spec.family, "role": spec.role, "prompt_mode": spec.prompt_mode,
        "multimodal": spec.multimodal,
        "n_agency_words": int((used.Category == "Agency").sum()),
        "n_communion_words": int((used.Category == "Communion").sum()),
        "n_duplicate_words": len(dups),
        "n_extra_duplicate_rows": int((dups.n_rows - 1).sum()) if len(dups) else 0,
    }
    return summary.assign(**meta_cols)[[*meta_cols, *summary.columns]], dups


def top_bottom(lp: pd.DataFrame, prompt: str, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-n and bottom-n tokens over the full vocab for one prompt."""
    ordered = (lp[["token_id", "token_decoded", prompt]]
               .sort_values([prompt, "token_id"], ascending=[False, True], kind="mergesort")
               .reset_index(drop=True))
    ranked = ordered.assign(rank=ordered.index + 1).rename(columns={prompt: "logprob"})
    return ranked.head(n), ranked.tail(n)
