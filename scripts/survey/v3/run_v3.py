"""Run PLAN_V3 generation: one model at a time, smallest first, gemma-4-E2B last.

Per model: [first-token CSV via generate.py if missing] -> sample.py -> verify -> delete only this
model's weights (same protected-cache rules as the survey). Appends to runs/v3/run_log.csv.

Usage: caffeinate -dimsu .venv/bin/python scripts/survey/v3/run_v3.py [--only ID ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi

SURVEY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SURVEY))
import run_survey as rs  # noqa: E402  (disk/cache helpers; CHILD_ENV disables shared blobs)

ROOT = SURVEY.parents[1]
RAW_DIR = ROOT / "runs" / "v3" / "raw"
LOG_DIR = ROOT / "runs" / "v3" / "logs"
RUN_LOG = ROOT / "runs" / "v3" / "run_log.csv"
SAMPLE = SURVEY / "v3" / "sample.py"
EXPECTED = 39 * 21
TIMEOUT_S = 4 * 3600


def mode_for(m: dict) -> str:
    return "chat_thinking" if m["prompt_mode"] == "chat_empty_think_prefill" else m["prompt_mode"]


def run_cmd(cmd: list[str], log: Path) -> int:
    try:
        with log.open("w") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=TIMEOUT_S,
                                  env=rs.CHILD_ENV).returncode
    except subprocess.TimeoutExpired:
        return -1


def verify_samples(path: Path) -> str:
    if not path.exists():
        return "raw jsonl missing"
    n = sum(1 for _ in path.open())
    return "" if n == EXPECTED else f"expected {EXPECTED} generations, got {n}"


def run_one(model: dict) -> dict:
    mid, safe = model["id"], model["id"].replace("/", "--")
    started, free0, notes, status = rs.now(), rs.disk_free_gb(), [], "ok"
    if not rs.out_path(mid).exists():
        rc = run_cmd([str(rs.PYTHON), str(rs.GENERATE), "--model", mid, "--mode", model["prompt_mode"],
                      "--out", str(rs.out_path(mid))], LOG_DIR / f"{safe}.firsttoken.log")
        problem = rs.verify_output(mid) if rc == 0 else f"generate.py exit {rc}"
        notes.append(f"first-token CSV: {problem or 'ok'}")
        status = "failed" if problem else status
    raw = RAW_DIR / f"{safe}.jsonl"
    rc = run_cmd([str(rs.PYTHON), str(SAMPLE), "--model", mid, "--mode", mode_for(model), "--out", str(raw)],
                 LOG_DIR / f"{safe}.sample.log")
    problem = verify_samples(raw) if rc == 0 else f"sample.py exit {rc}: {rs.tail(LOG_DIR / f'{safe}.sample.log')}"
    if problem:
        status = "failed"
        notes.append(problem)
    free_peak = rs.disk_free_gb()
    deleted, del_note = rs.delete_weights(mid)
    notes += [n for n in (del_note, f"disk free before delete {free_peak} GB") if n]
    return {"model": mid, "status": status, "started": started, "finished": rs.now(),
            "disk_free_gb_start": free0, "disk_free_gb_end": rs.disk_free_gb(),
            "weights_deleted": deleted, "note": " | ".join(notes)}


def ordered(only: list[str] | None) -> list[dict]:
    models = [m for m in yaml.safe_load(rs.CONFIG.read_text())["models"] if m.get("status") != "skipped"]
    if only:
        models = [m for m in models if m["id"] in set(only)]
    api = HfApi()
    size = {m["id"]: api.model_info(m["id"]).safetensors.total for m in models}
    return sorted(models, key=lambda m: (1 if m.get("run_last") else 0, size[m["id"]]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="*")
    p.add_argument("--exclude", nargs="*", default=[])
    a = p.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    models = [m for m in ordered(a.only) if m["id"] not in set(a.exclude)]
    print("order:", [m["id"] for m in models], flush=True)
    for m in models:
        print(f"[{rs.now()}] start {m['id']} (free {rs.disk_free_gb()} GB)", flush=True)
        row = run_one(m)
        new = not RUN_LOG.exists()
        with RUN_LOG.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[{rs.now()}] {row['status']} {m['id']}: {row['note']}", flush=True)


if __name__ == "__main__":
    main()
