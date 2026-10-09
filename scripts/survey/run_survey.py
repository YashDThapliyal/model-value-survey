"""Run the survey: one model at a time, smallest first, optional models last.

For each model: generate (downloads weights on first load) -> verify output CSV ->
delete ONLY that model's cache entry (never one listed in runs/protected_cache_entries.txt).
Appends one row per model to runs/run_log.csv. Run under caffeinate.

Usage: caffeinate -dimsu .venv/bin/python scripts/survey/run_survey.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils._shared_blobs import sweep_shared_blob, unreferenced_shared_blobs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "config" / "models.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "logprobs"
RUN_LOG = PROJECT_ROOT / "runs" / "run_log.csv"
LOG_DIR = PROJECT_ROOT / "runs" / "logs"
PROTECTED = PROJECT_ROOT / "runs" / "protected_cache_entries.txt"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
GENERATE = PROJECT_ROOT / "scripts" / "survey" / "generate.py"
N_PROMPTS = 54
N_META_COLS = 3
MIN_PROB_MASS = 0.98  # vocab rows may omit a few padded logits, so allow a little slack
TIMEOUT_S = 3 * 3600
MIN_FREE_GB = 3.0  # keep this much headroom beyond the model's weights
# huggingface_hub 1.x also keeps a cache-wide shared blob store; on this machine that
# stored every download twice and made per-repo deletion free nothing. Disable it for
# the generation subprocess so each model's files live only in its own cache entry.
CHILD_ENV = {**os.environ, "HF_HUB_DISABLE_SHARED_BLOBS": "1"}


@dataclass(frozen=True)
class RunRow:
    model: str
    status: str
    started: str
    finished: str
    disk_free_gb_start: float
    disk_free_gb_end: float
    weights_deleted: bool
    output: str
    note: str


def disk_free_gb() -> float:
    return round(shutil.disk_usage(HF_HUB_CACHE).free / 1e9, 2)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def cache_dir(model_id: str) -> Path:
    return Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"


def protected_entries() -> set[str]:
    return {line.strip() for line in PROTECTED.read_text().splitlines() if line.strip()}


def out_path(model_id: str) -> Path:
    return OUT_DIR / f"{model_id.replace('/', '--')}.csv"


def ordered_models(api: HfApi) -> list[dict]:
    models = [m for m in yaml.safe_load(CONFIG.read_text())["models"] if m.get("status") != "skipped"]

    def key(m: dict) -> tuple[int, int]:
        info = api.model_info(m["id"], files_metadata=False)
        params = info.safetensors.total if info.safetensors else 10**12
        return (1 if m.get("run_last") else 0, params)

    return sorted(models, key=key)


def verify_output(model_id: str) -> str:
    """Return '' if the CSV looks right, else a description of the problem."""
    path = out_path(model_id)
    if not path.exists():
        return "output CSV missing"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    df = pd.read_csv(path, keep_default_na=False)
    if df.shape[1] != N_META_COLS + N_PROMPTS:
        return f"expected {N_META_COLS + N_PROMPTS} columns, got {df.shape[1]}"
    if len(df) != meta["vocab_rows"]:
        return f"row count {len(df)} != vocab rows {meta['vocab_rows']}"
    lp = df.iloc[:, N_META_COLS:].to_numpy(dtype=float)
    if not np.isfinite(lp).all():
        return f"{int((~np.isfinite(lp)).sum())} non-finite log-probs"
    mass = np.exp(lp).sum(axis=0)
    if mass.min() < MIN_PROB_MASS or mass.max() > 1.0001:
        return f"probability mass per prompt out of range [{mass.min():.4f}, {mass.max():.4f}]"
    if df["token_id"].duplicated().any():
        return "duplicate token ids"
    return ""


def delete_weights(model_id: str) -> tuple[bool, str]:
    target = cache_dir(model_id)
    if target.name in protected_entries():
        return False, f"{target.name} is protected (pre-existing); not deleted"
    if not target.exists():
        return False, "no cache entry to delete"
    shutil.rmtree(target)
    swept = sweep_orphans_of(model_id)
    return True, f"swept {len(swept)} orphaned shared-store blobs" if swept else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sweep_orphans_of(model_id: str) -> list[str]:
    """Remove unreferenced shared-store blobs whose content is one of THIS model's LFS files.

    Content is matched by sha256 against the Hub's LFS metadata, so blobs belonging to
    anything else in the cache are never touched.
    """
    info = HfApi().model_info(model_id, files_metadata=True)
    hashes = {s.lfs.sha256: s.rfilename for s in info.siblings if s.lfs}
    sizes = {s.size for s in info.siblings if s.lfs}
    swept = []
    for path, size in unreferenced_shared_blobs(HF_HUB_CACHE).items():
        if size in sizes and _sha256(path) in hashes:
            sweep_shared_blob(path, cache_dir=HF_HUB_CACHE)
            swept.append(path.name)
    return swept


def needed_gb(api: HfApi, model_id: str) -> float:
    info = api.model_info(model_id, files_metadata=True)
    size = sum(
        (s.size or 0)
        for s in info.siblings
        if s.rfilename.endswith(".safetensors") and "/" not in s.rfilename
    )
    return size / 1e9


def run_one(api: HfApi, model: dict) -> RunRow:
    model_id, started, free0 = model["id"], now(), disk_free_gb()
    notes: list[str] = []
    status = "ok"
    if out_path(model_id).exists() and not verify_output(model_id):
        notes.append("output already present and verified (Gate C run); not regenerated")
    elif free0 < needed_gb(api, model_id) + MIN_FREE_GB:
        status, notes = "failed", [f"insufficient disk: {free0} GB free"]
    else:
        log = LOG_DIR / f"{model_id.replace('/', '--')}.log"
        cmd = [str(PYTHON), str(GENERATE), "--model", model_id, "--mode", model["prompt_mode"],
               "--out", str(out_path(model_id))]
        try:
            with log.open("w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=TIMEOUT_S,
                                    env=CHILD_ENV).returncode
        except subprocess.TimeoutExpired:
            rc = -1
            notes.append(f"timeout after {TIMEOUT_S}s")
        problem = verify_output(model_id) if rc == 0 else f"generate.py exit code {rc} (see {log.name})"
        if problem:
            status = "failed"
            notes.append(problem)
            if rc != 0:
                notes.append(tail(log))
    free_peak = disk_free_gb()
    deleted, del_note = delete_weights(model_id)
    if del_note:
        notes.append(del_note)
    notes.append(f"disk free before delete {free_peak} GB")
    return RunRow(model_id, status, started, now(), free0, disk_free_gb(), deleted,
                  str(out_path(model_id).relative_to(PROJECT_ROOT)), " | ".join(notes))


def tail(log: Path, n: int = 3) -> str:
    lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip() and "it/s" not in ln]
    return " / ".join(lines[-n:])[:600]


def append_row(row: RunRow) -> None:
    new = not RUN_LOG.exists()
    with RUN_LOG.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(row)))
        if new:
            writer.writeheader()
        writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="restrict to these model ids")
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    models = ordered_models(api)
    if args.only:
        models = [m for m in models if m["id"] in set(args.only)]
    print("order:", [m["id"] for m in models], flush=True)
    for model in models:
        print(f"[{now()}] start {model['id']} (free {disk_free_gb()} GB)", flush=True)
        row = run_one(api, model)
        append_row(row)
        print(f"[{now()}] {row.status} {model['id']}: {row.note}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
