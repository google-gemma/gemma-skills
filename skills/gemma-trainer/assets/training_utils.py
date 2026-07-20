#!/usr/bin/env python3
"""Dependency-free helpers shared by Gemma training templates."""

from typing import Any, Dict, List, Optional, Sequence


IGNORE_INDEX = -100


def to_conversational_preference(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string DPO fields to TRL's explicit conversational format."""
    fields = {
        "prompt": ("user", example["prompt"]),
        "chosen": ("assistant", example["chosen"]),
        "rejected": ("assistant", example["rejected"]),
    }
    converted = {}
    for name, (role, value) in fields.items():
        if isinstance(value, list):
            converted[name] = value
        elif isinstance(value, str) and value.strip():
            converted[name] = [{"role": role, "content": value}]
        else:
            raise ValueError(f"'{name}' must be a non-empty string or message list")
    return converted


def _find_subsequence(
    sequence: Sequence[int], needle: Sequence[int], start: int = 0
) -> Optional[int]:
    if not needle:
        raise ValueError("token marker cannot be empty")
    last_start = len(sequence) - len(needle)
    for index in range(start, last_start + 1):
        if list(sequence[index : index + len(needle)]) == list(needle):
            return index
    return None


def mask_assistant_tokens(
    input_ids: Sequence[int],
    assistant_start: Sequence[int],
    turn_end: Sequence[int],
    pad_token_id: Optional[int],
) -> List[int]:
    """Return labels containing every assistant span and masking all other tokens."""
    labels = [IGNORE_INDEX] * len(input_ids)
    search_from = 0
    found_assistant = False

    while True:
        header_index = _find_subsequence(input_ids, assistant_start, search_from)
        if header_index is None:
            break

        found_assistant = True
        content_start = header_index + len(assistant_start)
        end_index = _find_subsequence(input_ids, turn_end, content_start)
        content_end = len(input_ids) if end_index is None else end_index + len(turn_end)

        for index in range(content_start, content_end):
            token_id = input_ids[index]
            if pad_token_id is None or token_id != pad_token_id:
                labels[index] = token_id

        if end_index is None:
            break
        search_from = content_end

    if not found_assistant:
        raise ValueError("tokenized example does not contain an assistant turn")

    return labels


def processor_kwargs(texts: List[str], max_length: int) -> Dict[str, Any]:
    """Build processor arguments that enforce the configured training limit."""
    return {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": max_length,
    }
