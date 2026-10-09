#!/usr/bin/env python
"""Check Hub access, license, chat template, size and memory fit for candidate models.

Never downloads weights. The only file fetched is tokenizer_config.json, into a
throwaway temporary cache so the real HF cache is untouched.

Weight-size rule
----------------
* Only files at the repo root are counted (subfolders such as original/, onnx/,
  gguf/ or checkpoint dirs are ignored).
* Sum of root *.safetensors, excluding Mistral-style ``consolidated*`` duplicates.
* If (and only if) there are no root safetensors, sum root ``pytorch_model*.bin``
  / ``model*.bin`` files instead (never both, to avoid double counting).
* *.gguf, *.pt, *.pth, *.onnx are never counted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, constants, get_token, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)
from huggingface_hub.hf_api import ModelInfo

# ---- Memory-fit thresholds -------------------------------------------------
FALLBACK_TOTAL_RAM_BYTES = 18 * 1024**3
MPS_USABLE_FRACTION = 0.70  # macOS keeps the rest; MPS default high-watermark-ish
BYTES_PER_PARAM_HALF = 2  # bf16 / fp16
RUNTIME_OVERHEAD = 1.20  # activations, KV cache, framework buffers
TIGHT_FRACTION = 0.80  # above this share of the budget => "tight"
GB = 1e9

MULTIMODAL_PIPELINES = {"image-text-to-text", "any-to-any", "visual-question-answering"}
CHAT_TEMPLATE_FILES = {"chat_template.jinja", "chat_template.json"}
SIXTEEN_BIT_DTYPES = {"BF16", "F16"}


@dataclass(frozen=True)
class ModelEntry:
    id: str
    family: str = ""
    params: str = ""
    role: str = ""
    source: str = "main"  # "main" or "suggestion:<category>"


@dataclass(frozen=True)
class CheckResult:
    entry: ModelEntry
    exists: bool
    gated: str = "-"  # "False" / "auto" / "manual"
    access: str = "-"  # granted / not granted / no token / n/a
    license: str = "-"
    chat_template: str = "-"  # yes / no / unknown
    weight_bytes: int = 0
    weight_source: str = "-"
    param_count: int | None = None
    dtype: str = "-"
    pipeline: str = "-"
    fit: str = "-"
    est_mem_bytes: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accessible(self) -> bool:
        return self.exists and self.access in {"granted", "n/a"}

    @property
    def multimodal(self) -> bool:
        return self.pipeline in MULTIMODAL_PIPELINES


# ---- Config ----------------------------------------------------------------
def _to_entry(raw: dict[str, Any], source: str) -> ModelEntry:
    if not isinstance(raw, dict) or not raw.get("id"):
        raise ValueError(f"Invalid model entry in {source}: {raw!r}")
    return ModelEntry(
        id=str(raw["id"]),
        family=str(raw.get("family", "")),
        params=str(raw.get("params", "")),
        role=str(raw.get("role", "")),
        source=source,
    )


def load_entries(path: Path) -> list[ModelEntry]:
    data = yaml.safe_load(path.read_text()) or {}
    entries = [
        _to_entry(m, "main")
        for m in data.get("models") or []
        if m.get("status") != "skipped"
    ]
    for category, block in (data.get("suggestions") or {}).items():
        if not isinstance(block, dict):
            continue
        for m in block.get("models") or []:
            entries.append(_to_entry(m, f"suggestion:{category}"))
    return entries


# ---- Hub checks ------------------------------------------------------------
def gated_label(info: ModelInfo) -> str:
    return "False" if not info.gated else str(info.gated)


def check_access(api: HfApi, repo_id: str, gated: str, token: str | None) -> str:
    if gated == "False":
        return "n/a"
    if not token:
        return "no token"
    try:
        api.auth_check(repo_id, token=token)
        return "granted"
    except GatedRepoError:
        return "not granted"


def license_of(info: ModelInfo) -> str:
    card = info.card_data
    lic = card.get("license") if card else None
    if not lic:
        lic = next((t.split(":", 1)[1] for t in info.tags or [] if t.startswith("license:")), None)
    if lic == "other" and card and card.get("license_name"):
        lic = f"other ({card.get('license_name')})"
    return str(lic or "unknown")


def root_files(info: ModelInfo) -> list[tuple[str, int]]:
    return [(s.rfilename, s.size or 0) for s in info.siblings or [] if "/" not in s.rfilename]


def weight_size(info: ModelInfo) -> tuple[int, str]:
    files = root_files(info)
    st = [(n, s) for n, s in files if n.endswith(".safetensors") and not n.startswith("consolidated")]
    if st:
        return sum(s for _, s in st), f"safetensors x{len(st)}"
    bins = [
        (n, s)
        for n, s in files
        if n.endswith(".bin") and (n.startswith("pytorch_model") or n.startswith("model"))
    ]
    if bins:
        return sum(s for _, s in bins), f"bin x{len(bins)}"
    return 0, "none"


def has_chat_template(
    repo_id: str, info: ModelInfo, token: str | None, cache_dir: str
) -> tuple[str, str | None]:
    if any(n in CHAT_TEMPLATE_FILES for n, _ in root_files(info)):
        return "yes", None
    try:
        path = hf_hub_download(repo_id, "tokenizer_config.json", token=token, cache_dir=cache_dir)
        cfg = json.loads(Path(path).read_text())
        return ("yes" if cfg.get("chat_template") else "no"), None
    except EntryNotFoundError:
        return "no", None
    except GatedRepoError:
        return "unknown", "tokenizer_config.json gated"
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return "unknown", f"tokenizer_config: {type(exc).__name__}"


def param_info(info: ModelInfo) -> tuple[int | None, str]:
    st = info.safetensors
    if not st:
        return None, "-"
    dtypes = sorted(st.parameters, key=st.parameters.get, reverse=True)
    return st.total, "+".join(dtypes)


# ---- Memory fit ------------------------------------------------------------
def total_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return FALLBACK_TOTAL_RAM_BYTES


def mem_budget_bytes() -> int:
    return int(total_ram_bytes() * MPS_USABLE_FRACTION)


def estimate_mem(params: int | None, weight_bytes: int, dtype: str) -> int:
    if params:
        base = params * BYTES_PER_PARAM_HALF
    elif dtype in SIXTEEN_BIT_DTYPES or dtype == "-":
        base = weight_bytes  # assume 16-bit storage when metadata is missing
    else:
        return 0
    return int(base * RUNTIME_OVERHEAD)


def fit_label(est: int, budget: int) -> str:
    if est <= 0:
        return "unknown"
    if est <= budget * TIGHT_FRACTION:
        return "yes"
    return "tight" if est <= budget else "no"


# ---- Orchestration ---------------------------------------------------------
def check_model(api: HfApi, entry: ModelEntry, token: str | None, cache_dir: str) -> CheckResult:
    try:
        info = api.model_info(entry.id, files_metadata=True, token=token)
    except RepositoryNotFoundError:
        return CheckResult(entry=entry, exists=False, errors=("repo not found",))
    except Exception as exc:  # noqa: BLE001
        return CheckResult(entry=entry, exists=False, errors=(f"model_info: {exc!r}"[:200],))

    errors: list[str] = []
    gated = gated_label(info)
    try:
        access = check_access(api, entry.id, gated, token)
    except Exception as exc:  # noqa: BLE001
        access = "error"
        errors.append(f"auth_check: {type(exc).__name__}")
    chat, chat_err = has_chat_template(entry.id, info, token, cache_dir)
    if chat_err:
        errors.append(chat_err)
    wbytes, wsrc = weight_size(info)
    params, dtype = param_info(info)
    est = estimate_mem(params, wbytes, dtype)
    return CheckResult(
        entry=entry,
        exists=True,
        gated=gated,
        access=access,
        license=license_of(info),
        chat_template=chat,
        weight_bytes=wbytes,
        weight_source=wsrc,
        param_count=params,
        dtype=dtype,
        pipeline=info.pipeline_tag or "-",
        fit=fit_label(est, mem_budget_bytes()),
        est_mem_bytes=est,
        errors=tuple(errors),
    )


# ---- Reporting -------------------------------------------------------------
TABLE_HEADER = (
    "| Model | Source | Exists | Gated | Access | License | Chat tmpl | Weights GB "
    "| Params (B) | Dtype | Pipeline | Est. bf16 mem GB | Fits |"
)


def _row(r: CheckResult) -> str:
    params = f"{r.param_count / 1e9:.2f}" if r.param_count else "-"
    wgb = f"{r.weight_bytes / GB:.2f}" if r.weight_bytes else "-"
    mem = f"{r.est_mem_bytes / GB:.2f}" if r.est_mem_bytes else "-"
    pipe = f"{r.pipeline} (multimodal)" if r.multimodal else r.pipeline
    cells = [
        f"`{r.entry.id}`", r.entry.source, "yes" if r.exists else "NO", r.gated, r.access,
        r.license, r.chat_template, wgb, params, r.dtype, pipe, mem, r.fit,
    ]
    return "| " + " | ".join(cells) + " |"


def render_table(results: list[CheckResult]) -> str:
    sep = "|" + "|".join(["---"] * (TABLE_HEADER.count("|") - 1)) + "|"
    return "\n".join([TABLE_HEADER, sep, *(_row(r) for r in results)])


def free_disk_bytes() -> int:
    path = Path(constants.HF_HUB_CACHE)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _sum_accessible(results: list[CheckResult]) -> int:
    return sum(r.weight_bytes for r in results if r.accessible)


def render_summary(results: list[CheckResult], token: str | None) -> str:
    main = [r for r in results if r.entry.source == "main"]
    free = free_disk_bytes()
    main_gb, all_gb = _sum_accessible(main) / GB, _sum_accessible(results) / GB
    budget = mem_budget_bytes()
    need = [r for r in results if r.exists and r.access in {"no token", "not granted"}]
    missing = [r for r in results if not r.exists]
    multimodal = [r for r in results if r.multimodal]
    errored = [r for r in results if r.errors]
    lines = [
        "## Summary",
        "",
        f"- HF token present: {'yes' if token else 'no'}",
        f"- Memory budget: {total_ram_bytes() / 1024**3:.1f} GiB RAM x {MPS_USABLE_FRACTION} = "
        f"{budget / GB:.2f} GB; estimate = params x {BYTES_PER_PARAM_HALF} B x {RUNTIME_OVERHEAD}; "
        f"'tight' above {TIGHT_FRACTION:.0%} of budget.",
        f"- Disk for ACCESSIBLE main-list models: **{main_gb:.2f} GB**",
        f"- Disk for ACCESSIBLE main + suggestions: **{all_gb:.2f} GB**",
        f"- Disk if every existing model were accessible: "
        f"{sum(r.weight_bytes for r in results if r.exists) / GB:.2f} GB",
        f"- Free disk on HF cache volume ({str(constants.HF_HUB_CACHE).replace(str(Path.home()), '~', 1)}): **{free / GB:.2f} GB**",
        "",
        "### Models needing access requests",
        *([f"- `{r.entry.id}` (gated={r.gated}, {r.access})" for r in need] or ["- none"]),
        "",
        "### Nonexistent IDs",
        *([f"- `{r.entry.id}`" for r in missing] or ["- none"]),
        "",
        "### Notes",
        "- Weight size counts root-level *.safetensors only (bin fallback if none); "
        "subfolders (original/, onnx/, ...) and gguf files are ignored.",
        "- Multimodal models include vision/audio tower weights in the totals: "
        + (", ".join(f"`{r.entry.id}`" for r in multimodal) or "none") + ".",
    ]
    lines += [f"- `{r.entry.id}`: {'; '.join(r.errors)}" for r in errored]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=root / "config" / "models.yaml")
    p.add_argument("--output", type=Path, default=root / "access_report.md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entries = load_entries(args.config)
    token = get_token()
    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="hf_check_") as tmp_cache:
        results = [check_model(api, e, token, tmp_cache) for e in entries]
    report = "\n\n".join(["# Model access report", render_table(results), render_summary(results, token)])
    print(report)
    args.output.write_text(report + "\n")
    print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
