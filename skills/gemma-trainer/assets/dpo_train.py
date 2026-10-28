#!/usr/bin/env python3
"""Gemma Direct Preference Optimization (DPO) Script.

Aligns fine-tuned Gemma models with pairwise preference datasets.
Uses Unsloth (for maximum single-GPU VRAM optimization) or standard TRL
DPOTrainer.
"""

import argparse
import importlib
import logging
from typing import Any

from datasets import load_dataset
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemma-dpo")

# Try to import Unsloth before TRL/Transformers so monkey-patches take effect
try:
  importlib.import_module("unsloth").PatchDPOTrainer()
  HAS_UNSLOTH = True
  logger.info("Unsloth library detected! Fast DPOTrainer patched.")
except ImportError:
  HAS_UNSLOTH = False
  logger.info(
      "Unsloth not found. Falling back to standard HF PEFT + TRL DPOTrainer."
  )


def supports_bf16() -> bool:
  """Checks bf16 support on the active accelerator (CUDA or Apple Silicon MPS).

  torch.cuda.is_bf16_supported() only checks CUDA, so on MPS-only machines
  (e.g. Apple Silicon Macs) it always reports False and training silently
  falls back to fp16, which is more prone to NaN losses during QLoRA.

  Returns:
    True if bf16 is supported on the active accelerator, False otherwise.
  """
  if torch.cuda.is_available():
    return torch.cuda.is_bf16_supported()
  if torch.backends.mps.is_available():
    return torch.backends.mps.is_macos_or_newer(14, 0)
  return False


def load_model_and_processor(
    base_model_name: str,
    adapter_path: str,
    max_length: int,
    use_unsloth: bool,
    use_qlora: bool,
) -> tuple[Any, Any]:
  """Loads model and processor using either Unsloth or standard HF PEFT.

  Args:
    base_model_name: The original base model name or Hugging Face repo ID.
    adapter_path: Path to the existing SFT PEFT adapter directory.
    max_length: Maximum sequence length.
    use_unsloth: Whether to load using Unsloth optimizations.
    use_qlora: Whether to enable 4-bit QLoRA quantization.

  Returns:
    A tuple of `(model, processor)`.

  Raises:
    RuntimeError: If `use_unsloth` is True and Unsloth is not installed.
    ImportError: If `use_unsloth` is False and required Hugging Face multimodal
      dependencies (e.g. `torchvision`) cannot be imported.
  """
  if use_unsloth:
    if not HAS_UNSLOTH:
      raise RuntimeError("Unsloth is requested but not installed.")

    unsloth = importlib.import_module("unsloth")
    logger.info("Loading SFT adapter with Unsloth from %s...", adapter_path)
    # Unsloth handles loading the base model automatically from an adapter path
    model, processor = unsloth.FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_length,
        dtype=None,  # Auto-detection
        load_in_4bit=use_qlora,  # Enable QLoRA
        full_finetuning=False,
        use_gradient_checkpointing="unsloth",  # Saves immense VRAM
    )

    chat_templates = importlib.import_module("unsloth.chat_templates")
    processor = chat_templates.get_chat_template(
        processor, chat_template="gemma-4"
    )
  else:
    try:
      transformers = importlib.import_module("transformers")
      auto_model_cls = transformers.AutoModelForMultimodalLM
      auto_processor_cls = transformers.AutoProcessor
    except ImportError as e:
      raise ImportError(
          "Failed to import the Gemma 4 multimodal processor. This usually"
          " means `torchvision` is not installed (required by"
          " AutoProcessor/AutoModelForMultimodalLM even for text-only"
          " fine-tuning). Please run: pip install torchvision"
      ) from e
    peft = importlib.import_module("peft")

    torch_dtype = torch.bfloat16 if supports_bf16() else torch.float16

    model_kwargs: dict[str, Any] = dict(
        dtype=torch_dtype,  # What torch dtype to use
        device_map="auto",  # Let torch decide how to load the model
    )

    if use_qlora:
      model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch_dtype,
          bnb_4bit_quant_storage=torch_dtype,
      )

    logger.info(
        "Loading base model %s with HF transformers...", base_model_name
    )
    processor = auto_processor_cls.from_pretrained("google/gemma-4-E2B-it")
    model = auto_model_cls.from_pretrained(base_model_name, **model_kwargs)

    logger.info("Mounting trainable SFT adapter from %s...", adapter_path)
    model = peft.PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=True,
        adapter_name="default",
    )

  return model, processor


def train_dpo(
    base_model_name: str,
    adapter_path: str,
    dataset_path: str,
    test_size: float,
    output_dir: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    use_unsloth: bool,
    use_qlora: bool,
) -> None:
  """Unified training runner for DPO using DPOConfig."""
  trl = importlib.import_module("trl")
  model, processor = load_model_and_processor(
      base_model_name=base_model_name,
      adapter_path=adapter_path,
      max_length=max_length,
      use_unsloth=use_unsloth,
      use_qlora=use_qlora,
  )

  logger.info("Loading dataset from %s...", dataset_path)
  raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

  # Define a formatting function to inject the proper Chat Template structure
  def apply_dpo_template(example: dict[str, Any]) -> dict[str, str]:
    # Format the standalone user prompt using conversational dictionary lists
    prompt_msgs = [{"role": "user", "content": example["prompt"]}]
    chosen_msgs = prompt_msgs + [
        {"role": "assistant", "content": example["chosen"]}
    ]
    rejected_msgs = prompt_msgs + [
        {"role": "assistant", "content": example["rejected"]}
    ]

    # Extract the processed text strings using the model's processor/tokenizer
    return {
        "prompt": processor.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        ),
        "chosen": processor.apply_chat_template(chosen_msgs, tokenize=False),
        "rejected": processor.apply_chat_template(
            rejected_msgs, tokenize=False
        ),
    }

  logger.info("Formatting dataset strings with chat templates...")
  dataset = raw_dataset.map(apply_dpo_template).train_test_split(
      test_size=test_size
  )

  logger.info("Initializing DPOTrainer...")
  # Standard PEFT DPOTrainer can also set ref_model=None to save memory,
  # computing reference model outputs by disabling active LoRA adapters.
  trainer = trl.DPOTrainer(
      model=model,
      ref_model=None,
      processing_class=processor,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      args=trl.DPOConfig(
          output_dir=output_dir,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          num_train_epochs=epochs,
          learning_rate=learning_rate,
          bf16=supports_bf16(),
          fp16=not supports_bf16(),
          save_strategy="epoch",
          eval_strategy="epoch",
          max_length=max_length,
      ),
  )

  backend_label = "Unsloth" if use_unsloth else "standard HF"
  logger.info("Starting %s DPO training run...", backend_label)
  trainer_stats = trainer.train()
  logger.info("Training completed. Stats: %s", trainer_stats)

  # Save model/adapters
  logger.info("Saving preference-aligned adapters to %s...", output_dir)
  model.save_pretrained(output_dir)
  processor.save_pretrained(output_dir)
  logger.info("Done!")


def main() -> None:
  """Parses CLI arguments and runs DPO alignment training."""
  parser = argparse.ArgumentParser(
      description=(
          "Align Gemma models using Direct Preference Optimization (DPO)"
      )
  )

  # Path configuration
  parser.add_argument(
      "--base-model",
      type=str,
      default="google/gemma-4-E2B",
      help="The original base model name or HF repo",
  )
  parser.add_argument(
      "--adapter",
      type=str,
      required=True,
      help="Path to the existing SFT PEFT adapter directory",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      required=True,
      help="Path to JSON/JSONL preference dataset file",
  )
  parser.add_argument(
      "--output",
      type=str,
      default="./dpo_output",
      help="Output directory",
  )

  # Training configuration
  parser.add_argument(
      "--max-len", type=int, default=2048, help="Max sequence length"
  )
  parser.add_argument(
      "--test-size", type=float, default=0.2, help="dataset test split size"
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=1,
      help="Batch size per GPU (set to 1 to reduce OOM)",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=1,
      help="Training epochs (usually 1 is sufficient)",
  )
  parser.add_argument(
      "--lr",
      type=float,
      default=1e-6,
      help="DPO learning rate (typically much smaller than SFT, e.g., 1e-6)",
  )

  # Execution options
  parser.add_argument(
      "--force-hf",
      action="store_true",
      help="Force HF standard training even if Unsloth is installed",
  )
  parser.add_argument(
      "--force-no-qlora", action="store_true", help="Disable QLoRA"
  )

  args = parser.parse_args()

  use_unsloth = HAS_UNSLOTH and not args.force_hf
  use_qlora = not args.force_no_qlora

  train_dpo(
      base_model_name=args.base_model,
      adapter_path=args.adapter,
      dataset_path=args.dataset,
      test_size=args.test_size,
      output_dir=args.output,
      max_length=args.max_len,
      batch_size=args.batch_size,
      epochs=args.epochs,
      learning_rate=args.lr,
      use_unsloth=use_unsloth,
      use_qlora=use_qlora,
  )


if __name__ == "__main__":
  main()
