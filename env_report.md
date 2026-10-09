# Environment Report

Generated 2026-10-07, before any model or dataset downloads for this project.

| Item | Value |
|---|---|
| Machine | MacBook Pro (Mac15,7) |
| macOS | 26.2 (build 25C56) |
| Chip | Apple M3 Pro, 12 cores (6P + 6E). **Confirmed** |
| RAM | 18 GB unified (19,327,352,832 bytes). **Confirmed** |
| Free disk (data volume) | **~29 GB free** of 460 GB (93% used) |
| System Python | 3.14.2 (`/opt/homebrew/bin/python3`), no torch installed |
| Project venv | `.venv/`, Python 3.12.13 (created with `uv`; 3.14 is too new for reliable PyTorch wheels) |
| PyTorch (venv) | 2.14.1, MPS built: yes, MPS available: **yes** |
| transformers (venv) | 5.19.0 |
| huggingface_hub (venv) | 1.33.0 |
| Hugging Face login | **Not logged in.** `hf auth whoami` → "Not logged in". No `HF_*` env vars set. |
| HF cache location | `~/.cache/huggingface/hub` (default; `HF_HOME` not set) |
| HF cache size | 7.1 GB (hub), 40 KB (xet) |

## Existing HF cache contents

- datasets: `Adorg/ToolBench`, `ScaleAI/mrt`
- models: `answerdotai/ModernBERT-large`, `convaiinnovations/laya`, `gpt2`,
  `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen3.5-4B`

## Notes

- **Disk is the binding constraint**, not RAM. About 29 GB free, and the 7.1 GB HF cache
  holds unrelated models that could be cleared (`hf cache scan` / `hf cache delete`).
- **Not logged in** means gated repos (Meta Llama, Google Gemma) can't be downloaded.
  Run `hf auth login` in the venv (`! .venv/bin/hf auth login`) with a read token, then
  request access on each gated model page.
- The legacy `huggingface-cli` on PATH comes from the old system Python 3.9 framework
  install. Use `.venv/bin/hf` instead.
- MPS gives Metal about 70% of unified memory by default (~12–13 GB usable), so 4B models at
  bf16 (~8 GB) fit, but 7B+ would be tight.
