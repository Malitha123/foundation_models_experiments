"""Data and evaluation checks that do not require a GPU or model download."""

from __future__ import annotations

import random
import re
from collections import defaultdict


SYSTEM_MESSAGE = "You are a helpful medical assistant."


def _first_text(example, keys, default=None):
    for key in keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def map_dataset_fields(example):
    return {
        "instruction": _first_text(example, ("instruction", "question", "prompt", "task")),
        "input": _first_text(example, ("input", "context", "passage", "history"), ""),
        "output": _first_text(example, ("output", "response", "answer", "target", "completion")),
    }


def clean_records(records):
    """Normalize, reject blank examples, and remove exact duplicates before splitting."""
    cleaned, seen = [], set()
    invalid = duplicates = 0
    for example in records:
        row = map_dataset_fields(example)
        if row["instruction"] is None or row["output"] is None:
            invalid += 1
            continue
        key = (row["instruction"], row["input"], row["output"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        cleaned.append(row)
    return cleaned, {"invalid": invalid, "duplicates": duplicates}


def split_records(records, train_ratio=0.9, seed=42):
    """Keep all answers to the same prompt in one split to prevent leakage."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1, exclusive")
    groups = defaultdict(list)
    for row in records:
        groups[(row["instruction"], row["input"])].append(row)
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError("At least two distinct prompts are required for train/validation splitting")
    random.Random(seed).shuffle(keys)
    n_train = min(len(keys) - 1, max(1, int(len(keys) * train_ratio)))
    train = [row for key in keys[:n_train] for row in groups[key]]
    validation = [row for key in keys[n_train:] for row in groups[key]]
    return train, validation


def user_text(row):
    text = row["instruction"]
    if row.get("input"):
        text += f"\n\nInput: {row['input']}"
    return text


def render_chat(tokenizer, text, system_text=SYSTEM_MESSAGE, assistant_text=None):
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": text}]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=assistant_text is None,
    )


def has_training_completion(row, tokenizer, max_length):
    """Reject examples whose prompt consumes the entire truncated sequence."""
    prompt = render_chat(tokenizer, user_text(row))
    completed = render_chat(tokenizer, user_text(row), assistant_text=row["output"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(completed, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template's training prefix does not match its generation prefix")
    return len(prompt_ids) < min(max_length, len(full_ids))


def decode_new_tokens(generated_ids, input_ids, tokenizer):
    prompt_tokens = input_ids.shape[-1]
    return tokenizer.decode(generated_ids[0, prompt_tokens:], skip_special_tokens=True).strip()


def parse_judge_response(text, label="SCORE"):
    """Accept only an integer 1..5, preserving multiline reasoning."""
    if not isinstance(text, str) or not text.strip():
        return None, "Judge returned no text"
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*([1-5])\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return None, f"Invalid {label} response: {text.strip()}"
    reasoning = re.search(r"^\s*REASONING\s*:\s*(.*)", text, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    return int(match.group(1)), reasoning.group(1).strip() if reasoning else text.strip()


def hub_publish_target(config, repo_id):
    """A repository name alone must never enable publishing."""
    if not config.get("push_to_hub", False):
        return None
    if not repo_id or not repo_id.strip():
        raise ValueError("Set HF_REPO_ID when push_to_hub=True")
    return repo_id.strip()
