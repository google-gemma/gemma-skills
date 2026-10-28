#!/usr/bin/env python3
"""Gemma Dataset Preparation & Validation Tool.

This script provides utilities and a CLI interface to load, parse, format,
and validate datasets for Supervised Fine-Tuning (SFT), Direct Preference
Optimization (DPO), and Reward Modeling with Gemma models.
"""

import argparse
from collections.abc import Mapping
import importlib
import json
import logging
import os
import sys
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemma-dataset-prep")


def load_dataset_file(file_path: str) -> list[dict[str, Any]]:
  """Loads a dataset from a JSON or JSONL file."""
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

  data = []
  _, ext = os.path.splitext(file_path.lower())

  if ext == ".jsonl":
    with open(file_path, "r", encoding="utf-8") as f:
      for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
          continue
        try:
          data.append(json.loads(line))
        except json.JSONDecodeError as e:
          logger.error("Invalid JSON on line %d: %s", i, e)
          raise
  elif ext == ".json":
    with open(file_path, "r", encoding="utf-8") as f:
      try:
        raw_data = json.load(f)
        if isinstance(raw_data, list):
          data = raw_data
        else:
          data = [raw_data]
      except json.JSONDecodeError as e:
        logger.error("Invalid JSON file: %s", e)
        raise
  else:
    raise ValueError("Unsupported file format. Please use .json or .jsonl")

  logger.info("Successfully loaded %d items from %s", len(data), file_path)
  return data


_VALID_CHAT_ROLES = ("system", "user", "assistant", "tool")


def _validate_chat_messages(messages: Any, idx: int, field_name: str) -> bool:
  """Validates a non-empty list of chat message dictionaries."""
  if not isinstance(messages, list) or not messages:
    logger.warning("Item %d: '%s' must be a non-empty list.", idx, field_name)
    return False

  for m_idx, message in enumerate(messages):
    if not isinstance(message, Mapping):
      logger.warning(
          "Item %d, %s[%d]: Message must be a dictionary.",
          idx,
          field_name,
          m_idx,
      )
      return False
    if "role" not in message or "content" not in message:
      logger.warning(
          "Item %d, %s[%d]: Message must contain 'role' and 'content' keys.",
          idx,
          field_name,
          m_idx,
      )
      return False
    if message["role"] not in _VALID_CHAT_ROLES:
      logger.warning(
          "Item %d, %s[%d]: Role must be 'system', 'user', 'assistant', or"
          " 'tool' (Standard chat roles).",
          idx,
          field_name,
          m_idx,
      )
      return False

  return True


def _validate_sft_prompt_completion(item: Mapping[str, Any], idx: int) -> bool:
  """Validates an SFT item in prompt-completion format."""
  if "prompt" not in item or "completion" not in item:
    logger.warning(
        "Item %d: Prompt-completion SFT requires both 'prompt' and"
        " 'completion' fields.",
        idx,
    )
    return False

  prompt = item["prompt"]
  completion = item["completion"]

  if isinstance(prompt, list) and isinstance(completion, list):
    if not _validate_chat_messages(prompt, idx, "prompt"):
      return False
    if not _validate_chat_messages(completion, idx, "completion"):
      return False
    if (
        prompt[-1]["role"] == "assistant"
        and completion[0]["role"] == "assistant"
    ):
      logger.warning(
          "Item %d: 'prompt' ends with 'assistant' and 'completion' starts"
          " with 'assistant'. Prompt turns should end with 'user' or 'tool'.",
          idx,
      )
      return False
    if completion[-1]["role"] != "assistant":
      logger.warning(
          "Item %d: 'completion' does not end with an 'assistant' message."
          " SFT models train on generating the assistant completion.",
          idx,
      )
      return False
    return True

  if isinstance(prompt, str) and isinstance(completion, str):
    if not prompt.strip() or not completion.strip():
      logger.warning(
          "Item %d: String 'prompt' and 'completion' fields must be non-empty.",
          idx,
      )
      return False
    return True

  logger.warning(
      "Item %d: 'prompt' and 'completion' must both be non-empty lists of"
      " role/content dictionaries or both be non-empty strings.",
      idx,
  )
  return False


def validate_sft_item(item: Mapping[str, Any], idx: int) -> bool:
  """Validates an SFT item in conversational or prompt-completion format."""
  has_messages = "messages" in item
  has_prompt_completion = "prompt" in item or "completion" in item

  if has_messages and has_prompt_completion:
    logger.warning(
        "Item %d: Ambiguous SFT format. Provide either 'messages' or"
        " 'prompt' and 'completion', not both.",
        idx,
    )
    return False

  if has_messages:
    messages = item["messages"]
    if not _validate_chat_messages(messages, idx, "messages"):
      return False
    # SFT datasets should end with an 'assistant' turn so the model learns what
    # to generate
    if messages[-1]["role"] != "assistant":
      logger.warning(
          "Item %d: The conversation does not end with an 'assistant' message."
          " SFT models train on generating the final 'assistant' content.",
          idx,
      )
      return False
    return True

  if has_prompt_completion:
    return _validate_sft_prompt_completion(item, idx)

  logger.warning(
      "Item %d: Missing SFT fields. Expected either 'messages' or both"
      " 'prompt' and 'completion'.",
      idx,
  )
  return False


def validate_sft_prompt_completion_tokenization(
    item: Mapping[str, Any], idx: int, tokenizer: Any
) -> bool:
  """Validates prompt prefix token alignment for prompt-completion SFT items.

  Verifies that standalone prompt tokens match the prefix of the combined
  prompt+completion token sequence so TRL masks prompt loss accurately.

  Args:
    item: Dictionary containing 'prompt' and 'completion' fields.
    idx: Index of the item in the dataset for logging.
    tokenizer: Hugging Face tokenizer instance supporting chat templates.

  Returns:
    True if the standalone prompt token prefix matches the combined sequence,
    False otherwise.
  """
  try:
    if isinstance(item["prompt"], list):
      prompt_str = tokenizer.apply_chat_template(
          item["prompt"], tokenize=False, add_generation_prompt=True
      )
      full_str = tokenizer.apply_chat_template(
          item["prompt"] + item["completion"], tokenize=False
      )
    else:
      prompt_str = item["prompt"]
      full_str = f"{item['prompt']}{item['completion']}"

    prompt_ids = tokenizer.encode(prompt_str)
    full_ids = tokenizer.encode(full_str)
    full_prefix = full_ids[: len(prompt_ids)]

    if prompt_ids != full_prefix:
      logger.warning(
          "Item %d: Token mismatch between standalone prompt and"
          " prompt+completion sequence prefix.\nPrompt IDs:  %s\nCombined"
          " IDs:%s",
          idx,
          prompt_ids,
          full_prefix,
      )
      logger.warning("Item %d Prompt Raw: %r", idx, prompt_str)
      logger.warning("Item %d Combined Raw: %r", idx, full_str)
      return False

    return True
  except (KeyError, TypeError, ValueError, AttributeError, RuntimeError) as e:
    logger.error("Item %d: Exception during token validation: %s", idx, e)
    return False


def _format_sft_sample_for_length(
    item: Mapping[str, Any], tokenizer: Any | None
) -> str:
  """Formats a validated SFT item into text for sequence length estimation."""
  if "messages" in item:
    messages = item["messages"]
  elif isinstance(item["prompt"], list):
    messages = item["prompt"] + item["completion"]
  else:
    return f"{item['prompt']}{item['completion']}"

  if tokenizer is not None:
    return tokenizer.apply_chat_template(messages, tokenize=False)
  return "\n".join(str(msg.get("content", "")) for msg in messages)


def validate_dpo_item(item: Mapping[str, Any], idx: int) -> bool:
  """Validates a DPO pairwise preference item."""
  required_keys = ["prompt", "chosen", "rejected"]
  missing = [k for k in required_keys if k not in item]
  if missing:
    logger.warning(
        "Item %d: Missing preference keys: %s. DPO requires 'prompt',"
        " 'chosen', and 'rejected'.",
        idx,
        missing,
    )
    return False

  for key in required_keys:
    val = item[key]
    if not isinstance(val, str) or not val.strip():
      logger.warning(
          "Item %d: Field '%s' must be a non-empty string.", idx, key
      )
      return False

  if item["chosen"] == item["rejected"]:
    logger.warning(
        "Item %d: 'chosen' and 'rejected' strings are identical. Model will"
        " not learn preferences.",
        idx,
    )
    return False

  return True


def validate_dpo_tokenization(
    item: Mapping[str, Any], idx: int, tokenizer: Any
) -> bool:
  """Validates a DPO preference item to ensure tokenization consistency.

  Verifies that the standalone prompt tokens perfectly match the beginning
  of both the prompt+chosen and prompt+rejected token sequences.

  Args:
    item: Dictionary containing 'prompt', 'chosen', and 'rejected' fields.
    idx: Index of the item in the dataset for logging.
    tokenizer: Hugging Face tokenizer instance supporting chat templates.

  Returns:
    True if the standalone prompt token prefix matches both sequences, False
    otherwise.
  """
  try:
    # 1. Structure the components properly into standard chat formats if
    # needed. This assumes your prompt doesn't already contain chat markers.
    prompt_chat = [{"role": "user", "content": item["prompt"]}]

    # Hugging Face TRL library style: 'chosen' and 'rejected' are treated as
    # the assistant's response.
    chosen_chat = prompt_chat + [
        {"role": "assistant", "content": item["chosen"]}
    ]
    rejected_chat = prompt_chat + [
        {"role": "assistant", "content": item["rejected"]}
    ]

    # 2. Apply chat template without tokenizing to see raw strings
    prompt_str = tokenizer.apply_chat_template(
        prompt_chat, tokenize=False, add_generation_prompt=True
    )
    chosen_str = tokenizer.apply_chat_template(chosen_chat, tokenize=False)
    rejected_str = tokenizer.apply_chat_template(rejected_chat, tokenize=False)

    # 3. Tokenize all strings
    prompt_ids = tokenizer.encode(prompt_str)
    chosen_ids = tokenizer.encode(chosen_str)
    rejected_ids = tokenizer.encode(rejected_str)

    prompt_len = len(prompt_ids)

    # 4. CRITICAL CHECK: Does the prompt prefix match the combined prefixes
    # exactly?
    chosen_prefix = chosen_ids[:prompt_len]
    rejected_prefix = rejected_ids[:prompt_len]

    mismatch_found = False

    if prompt_ids != chosen_prefix:
      logger.warning(
          "Item %d: Token mismatch between standalone prompt and 'chosen'"
          " sequence prefix.\nPrompt IDs:  %s\nChosen IDs:  %s",
          idx,
          prompt_ids,
          chosen_prefix,
      )
      mismatch_found = True

    if prompt_ids != rejected_prefix:
      logger.warning(
          "Item %d: Token mismatch between standalone prompt and 'rejected'"
          " sequence prefix.\nPrompt IDs:  %s\nRejected IDs:%s",
          idx,
          prompt_ids,
          rejected_prefix,
      )
      mismatch_found = True

    if mismatch_found:
      # Provide debug context so you can spot whitespace or special token
      # issues
      logger.warning("Item %d Prompt Raw: %r", idx, prompt_str)
      logger.warning("Item %d Chosen Raw: %r", idx, chosen_str)
      return False

    return True

  except (KeyError, TypeError, ValueError, AttributeError, RuntimeError) as e:
    logger.error("Item %d: Exception during token validation: %s", idx, e)
    return False


def run_validation(
    file_path: str,
    task_type: str,
    max_seq_length: int = 2048,
    tokenizer_name: str | None = None,
) -> bool:
  """Runs a complete validation suite over the dataset.

  Args:
    file_path: Path to the JSON or JSONL dataset file.
    task_type: Target training task ('sft', 'dpo', or 'reward').
    max_seq_length: Maximum allowed sequence length in tokens.
    tokenizer_name: Optional Hugging Face tokenizer ID. If None, skips
      tokenization checks and uses a character count heuristic.

  Returns:
    True if all dataset items pass schema validation, False otherwise.
  """
  logger.info(
      "Starting validation for task: '%s' on file: '%s'", task_type, file_path
  )

  try:
    dataset = load_dataset_file(file_path)
  except (OSError, ValueError) as e:
    logger.error("Failed to load file: %s", e)
    return False

  valid_count = 0
  total_count = len(dataset)

  # Try importing transformers tokenizer if provided
  tokenizer = None
  if tokenizer_name:
    try:
      transformers = importlib.import_module("transformers")
      logger.info("Loading tokenizer '%s' for length checks...", tokenizer_name)
      tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
      logger.warning(
          "transformers library not installed. Skipping precise token length"
          " checks (will use character count heuristic)."
      )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
      logger.error("Could not load tokenizer '%s': %s", tokenizer_name, e)

  long_sequences = 0

  for i, item in enumerate(dataset):
    is_valid = False
    sample_text_for_length = ""

    if task_type == "sft":
      is_valid = validate_sft_item(item, i)
      if is_valid:
        if tokenizer is not None and "prompt" in item:
          is_token_consistent = validate_sft_prompt_completion_tokenization(
              item, i, tokenizer
          )
          if not is_token_consistent:
            logger.warning(
                "Item %d failed prompt-completion token alignment validation.",
                i,
            )
        sample_text_for_length = _format_sft_sample_for_length(item, tokenizer)
    elif task_type in ["dpo", "reward"]:
      is_valid = validate_dpo_item(item, i)
      if is_valid:
        if tokenizer is not None:
          # Run the strict token consistency validation
          is_token_consistent = validate_dpo_tokenization(item, i, tokenizer)
          if not is_token_consistent:
            # You can choose to mark the item invalid, or just keep it as a
            # warning
            logger.warning("Item %d failed token alignment validation.", i)

          # Safe sequence length measurement using the full text
          sample_text_for_length = tokenizer.apply_chat_template(
              [
                  {"role": "user", "content": item["prompt"]},
                  {"role": "assistant", "content": item["chosen"]},
              ],
              tokenize=False,
          )
        else:
          sample_text_for_length = f"{item['prompt']}\n{item['chosen']}"

    if is_valid:
      logger.info("Item %d: %s", i, sample_text_for_length)
      valid_count += 1

      # Check length constraints
      if tokenizer is not None:
        tokens = tokenizer.encode(sample_text_for_length)
        num_tokens = len(tokens)
      else:
        num_tokens = len(sample_text_for_length) // 4

      if num_tokens > max_seq_length:
        long_sequences += 1
        logger.warning(
            "Item %d: Estimated length (%d tokens) exceeds target"
            " max_seq_length (%d). This might trigger OOM or truncation.",
            i,
            num_tokens,
            max_seq_length,
        )

  success_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
  logger.info("-------------------- SUMMARY --------------------")
  logger.info("Total samples analyzed: %d", total_count)
  logger.info(
      "Valid samples for %s: %d (%.2f%%)",
      task_type.upper(),
      valid_count,
      success_rate,
  )
  logger.info(
      "Samples exceeding max length (%d): %d", max_seq_length, long_sequences
  )

  if valid_count == total_count and long_sequences == 0:
    logger.info("STATUS: SUCCESS. Dataset is fully valid and clean!")
    return True
  elif valid_count == total_count:
    logger.warning(
        "STATUS: PASSED with warnings. Schema is correct, but some samples"
        " exceed the recommended max token length."
    )
    return True
  else:
    logger.error(
        "STATUS: FAILED. Found %d invalid samples. Please clean dataset before"
        " training.",
        total_count - valid_count,
    )
    return False


def main() -> None:
  """Parses CLI arguments and runs dataset validation."""
  parser = argparse.ArgumentParser(
      description="Gemma Dataset Preparation & Validation CLI"
  )
  parser.add_argument(
      "--file",
      type=str,
      required=True,
      help="Path to JSON or JSONL file to validate",
  )
  parser.add_argument(
      "--task",
      type=str,
      choices=["sft", "dpo", "reward"],
      required=True,
      help="Target training task",
  )
  parser.add_argument(
      "--max-len",
      type=int,
      default=2048,
      help="Target maximum sequence length (default: 2048)",
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      default="google/gemma-4-E2B-it",
      help="HF Tokenizer name to run precise token validation",
  )

  args = parser.parse_args()

  success = run_validation(
      file_path=args.file,
      task_type=args.task,
      max_seq_length=args.max_len,
      tokenizer_name=args.tokenizer,
  )

  sys.exit(0 if success else 1)


if __name__ == "__main__":
  main()
