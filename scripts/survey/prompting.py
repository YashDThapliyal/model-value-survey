"""Prompt construction for the value survey.

Mirrors the authors' `tokenize_prompt` (vendor/.../scripts/generate_logprobs.py):
instruct models get `apply_chat_template([user msg], add_generation_prompt=True)` with no
system prompt; base models get the raw prompt. Adds three survey-specific modes, and pins
the date that chat templates inject via `strftime_now`.
"""

from __future__ import annotations

import datetime as _dt

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import chat_template_utils

FIXED_DATE = _dt.datetime(2024, 7, 26, 12, 0, 0)  # Llama 3.x template's own fallback date
EMPTY_THINK = "<think>\n\n</think>\n\n"
PROMPT_MODES = ("chat", "chat_no_think", "chat_empty_think_prefill", "raw")


class _FrozenDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102 - matches datetime.now signature
        return FIXED_DATE if tz is None else FIXED_DATE.replace(tzinfo=tz)


def freeze_template_date() -> None:
    """Make every chat template's `strftime_now(...)` render FIXED_DATE.

    transformers defines `strftime_now` as `datetime.now().strftime(fmt)` inside the template
    compiler, resolving `datetime` from the chat_template_utils module globals at call time.
    """
    chat_template_utils.datetime = _FrozenDatetime


def render_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str, mode: str) -> str:
    """Return the exact text fed to the model (for logging and checks)."""
    if mode == "raw":
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"enable_thinking": False} if mode == "chat_no_think" else {}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    if mode == "chat_empty_think_prefill":
        text = _prefill_empty_think(text)
    return text


def _prefill_empty_think(text: str) -> str:
    """Close the reasoning block so the next token is the answer.

    Newer DeepSeek-R1 distill templates already end the generation prompt with "<think>\\n";
    in that case only the closing part is appended.
    """
    if text.endswith("<think>\n"):
        return text + "\n</think>\n\n"
    return text + EMPTY_THINK


def build_inputs(
    tokenizer: PreTrainedTokenizerBase, prompt: str, mode: str
) -> dict[str, torch.Tensor]:
    """Tokenize a prompt the way the authors do, extended with the survey modes."""
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {mode}")
    if mode == "chat":
        # Identical call to the authors' instruction-tuned path.
        return dict(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        )
    if mode == "raw":
        # Identical call to the authors' pretrained path.
        return dict(tokenizer(prompt, return_tensors="pt"))
    # Template text already contains any BOS token, so do not add special tokens again.
    text = render_prompt(tokenizer, prompt, mode)
    return dict(tokenizer(text, return_tensors="pt", add_special_tokens=False))
