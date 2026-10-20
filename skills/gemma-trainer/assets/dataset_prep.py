#!/usr/bin/env python3
"""
Gemma Dataset Preparation & Validation Tool
This script provides utilities and a CLI interface to load, parse, format,
and validate datasets for Supervised Fine-Tuning (SFT), Direct Preference
Optimization (DPO), and Reward Modeling with Gemma models.
"""

import os
import argparse
import json
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gemma-dataset-prep")

def load_dataset_file(file_path: str) -> List[Dict[str, Any]]:
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
                    logger.error(f"Invalid JSON on line {i}: {e}")
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
                logger.error(f"Invalid JSON file: {e}")
                raise
    else:
        raise ValueError("Unsupported file format. Please use .json or .jsonl")
        
    logger.info(f"Successfully loaded {len(data)} items from {file_path}")
    return data

def validate_sft_item(item: Dict[str, Any], idx: int) -> bool:
    """Validates an SFT conversation item."""
    if not isinstance(item, dict):
        logger.warning(f"Item {idx}: SFT samples must be dictionaries.")
        return False
    if "messages" not in item:
        logger.warning(f"Item {idx}: Missing 'messages' field. SFT expects a list of role/content dictionaries.")
        return False
    
    messages = item["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        logger.warning(f"Item {idx}: 'messages' must be a non-empty list.")
        return False
        
    for m_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            logger.warning(f"Item {idx}, Message {m_idx}: Message must be a dictionary.")
            return False
        if "role" not in msg or "content" not in msg:
            logger.warning(f"Item {idx}, Message {m_idx}: Message must contain 'role' and 'content' keys.")
            return False
        if msg["role"] not in ["system", "user", "assistant", "tool"]:
            logger.warning(f"Item {idx}, Message {m_idx}: Role must be 'system', 'user', 'assistant', or 'tool' (Standard chat roles).")
            return False
            
    # SFT datasets should end with a 'assistant' turn so the model learns what to generate
    if messages[-1]["role"] != "assistant":
        logger.warning(f"Item {idx}: The conversation does not end with a 'assistant' message. SFT models train on generating the final 'assistant' content.")
        return False
    final_content = messages[-1]["content"]
    if isinstance(final_content, str):
        has_response = bool(final_content.strip())
    elif isinstance(final_content, list):
        has_response = any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"].strip()
            for block in final_content
        )
    else:
        has_response = False
    if not has_response:
        logger.warning(f"Item {idx}: The final assistant message needs non-empty text content.")
        return False
        
    return True

def validate_dpo_item(item: Dict[str, Any], idx: int) -> bool:
    """Validates a DPO pairwise preference item."""
    if not isinstance(item, dict):
        logger.warning(f"Item {idx}: Preference samples must be dictionaries.")
        return False
    required_keys = ["prompt", "chosen", "rejected"]
    missing = [k for k in required_keys if k not in item]
    if missing:
        logger.warning(f"Item {idx}: Missing preference keys: {missing}. DPO requires 'prompt', 'chosen', and 'rejected'.")
        return False
        
    for key in required_keys:
        val = item[key]
        if not isinstance(val, str) or not val.strip():
            logger.warning(f"Item {idx}: Field '{key}' must be a non-empty string.")
            return False
            
    if item["chosen"] == item["rejected"]:
        logger.warning(f"Item {idx}: 'chosen' and 'rejected' strings are identical. Model will not learn preferences.")
        return False
        
    return True

def validate_dpo_tokenization(item: Dict[str, Any], idx: int, tokenizer: Any) -> bool:
    """
    Validates a DPO preference item to ensure tokenization consistency.
    Verifies that the standalone prompt tokens perfectly match the beginning
    of both the prompt+chosen and prompt+rejected token sequences.
    """
    try:
        # 1. Structure the components properly into standard chat formats if needed.
        # This assumes your prompt doesn't already contain chat markers.
        prompt_chat = [{"role": "user", "content": item["prompt"]}]

        # Hugging Face TRL library style: 'chosen' and 'rejected' are treated as the assistant's response.
        chosen_chat = prompt_chat + [{"role": "assistant", "content": item["chosen"]}]
        rejected_chat = prompt_chat + [{"role": "assistant", "content": item["rejected"]}]

        # 2. Apply the tokenizer's template directly. Rendering and then calling encode()
        # can add special tokens twice and create a false boundary mismatch.
        prompt_ids = tokenizer.apply_chat_template(
            prompt_chat, tokenize=True, add_generation_prompt=True)
        chosen_ids = tokenizer.apply_chat_template(chosen_chat, tokenize=True)
        rejected_ids = tokenizer.apply_chat_template(rejected_chat, tokenize=True)

        prompt_len = len(prompt_ids)

        # 4. CRITICAL CHECK: Does the prompt prefix match the combined prefixes exactly?
        chosen_prefix = chosen_ids[:prompt_len]
        rejected_prefix = rejected_ids[:prompt_len]

        mismatch_found = False

        if prompt_ids != chosen_prefix:
            logger.warning(f"Item {idx}: Token mismatch in the chosen-sequence prompt prefix.")
            mismatch_found = True

        if prompt_ids != rejected_prefix:
            logger.warning(f"Item {idx}: Token mismatch in the rejected-sequence prompt prefix.")
            mismatch_found = True

        if mismatch_found:
            return False

        return True

    except Exception as e:
        logger.error(
            f"Item {idx}: Token validation failed with {type(e).__name__}; "
            "sample contents were omitted."
        )
        return False

def run_validation(file_path: str, task_type: str, max_seq_length: int = 2048, tokenizer_name: str = None) -> bool:
    """Runs a complete validation suite over the dataset."""
    logger.info(f"Starting validation for task: '{task_type}' on file: '{file_path}'")
    
    try:
        dataset = load_dataset_file(file_path)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return False
        
    if max_seq_length <= 0:
        logger.error("max_seq_length must be positive.")
        return False

    valid_count = 0
    total_count = len(dataset)
    if total_count == 0:
        logger.error("Dataset is empty; refusing to report a vacuous success.")
        return False
    
    # Try importing transformers tokenizer if provided
    if not tokenizer_name:
        logger.error("A tokenizer is required for template, boundary, and length validation.")
        return False
    try:
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer '{tokenizer_name}' for template and length checks...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except ImportError:
        logger.error("transformers is required for tokenizer-aware validation.")
        return False
    except Exception as e:
        logger.error(f"Could not load tokenizer '{tokenizer_name}': {type(e).__name__}")
        return False
            
    long_sequences = 0
    
    for i, item in enumerate(dataset):
        is_valid = False
        token_lengths = []
        
        if task_type == "sft":
            is_valid = validate_sft_item(item, i)
            if is_valid:
                try:
                    token_lengths.append(len(tokenizer.apply_chat_template(
                        item["messages"], tokenize=True)))
                except Exception as e:
                    logger.error(
                        f"Item {i}: Template validation failed with {type(e).__name__}; "
                        "sample contents were omitted."
                    )
                    is_valid = False
        elif task_type in ["dpo", "reward"]:
            is_valid = validate_dpo_item(item, i)
            if is_valid:
                # Run the strict token consistency validation
                is_token_consistent = validate_dpo_tokenization(item, i, tokenizer)
                if not is_token_consistent:
                    logger.warning(f"Item {i} failed token alignment validation.")
                    is_valid = False

                # Both branches are training inputs and must fit without truncation.
                try:
                    for response_key in ("chosen", "rejected"):
                        tokens = tokenizer.apply_chat_template(
                            [
                                {"role": "user", "content": item["prompt"]},
                                {"role": "assistant", "content": item[response_key]},
                            ],
                            tokenize=True,
                        )
                        token_lengths.append(len(tokens))
                except Exception as e:
                    logger.error(
                        f"Item {i}: Template validation failed with {type(e).__name__}; "
                        "sample contents were omitted."
                    )
                    is_valid = False
                
        if is_valid:
            valid_count += 1
            
            # Check length constraints
            num_tokens = max(token_lengths)
            if num_tokens > max_seq_length:
                long_sequences += 1
                logger.warning(f"Item {i}: Estimated length ({num_tokens} tokens) exceeds target max_seq_length ({max_seq_length}). This might trigger OOM or truncation.")

    success_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
    logger.info("-------------------- SUMMARY --------------------")
    logger.info(f"Total samples analyzed: {total_count}")
    logger.info(f"Valid samples for {task_type.upper()}: {valid_count} ({success_rate:.2f}%)")
    logger.info(f"Samples exceeding max length ({max_seq_length}): {long_sequences}")
    
    if valid_count == total_count and long_sequences == 0:
        logger.info("STATUS: SUCCESS. Dataset is fully valid and clean!")
        return True
    elif valid_count == total_count:
        logger.error("STATUS: FAILED. One or more samples exceed max_seq_length and would be truncated.")
        return False
    else:
        logger.error(f"STATUS: FAILED. Found {total_count - valid_count} invalid samples. Please clean dataset before training.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Dataset Preparation & Validation CLI")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON or JSONL file to validate")
    parser.add_argument("--task", type=str, choices=["sft", "dpo", "reward"], required=True, help="Target training task")
    parser.add_argument("--max-len", type=int, default=2048, help="Target maximum sequence length (default: 2048)")
    parser.add_argument("--tokenizer", type=str, default="google/gemma-4-E2B-it", help="HF Tokenizer name to run precise token validation")
    
    args = parser.parse_args()
    
    success = run_validation(
        file_path=args.file,
        task_type=args.task,
        max_seq_length=args.max_len,
        tokenizer_name=args.tokenizer
    )
    
    import sys
    sys.exit(0 if success else 1)
