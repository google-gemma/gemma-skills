#!/usr/bin/env python3
"""Gemma Reward Modeling Script.

Trains a classification-style reward head on top of a Gemma base model.
Uses Hugging Face PEFT + TRL RewardTrainer.

NOTE: AutoModelForSequenceClassification doesn't support Gemma 4 yet.
      https://github.com/huggingface/transformers/issues/45373
"""

import argparse
import logging

import datasets
import peft
import torch
import transformers
import trl

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemma-reward-model")


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


def train_reward_model(
    model_name: str,
    dataset_path: str,
    test_size: float,
    output_dir: str,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    use_qlora: bool,
) -> None:
  """Fine-tunes a Gemma sequence classification model as a Reward Model."""
  torch_dtype = torch.bfloat16 if supports_bf16() else torch.float16

  model_kwargs = dict(
      dtype=torch_dtype,  # What torch dtype to use
      device_map="auto",  # Let torch decide how to load the model
      num_labels=1,  # sequence classification (1 label)
  )

  if use_qlora:
    model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

  # Load model & processor
  logger.info(
      "Loading %s with classification head (num_labels=1)...", model_name
  )
  processor = transformers.AutoProcessor.from_pretrained(
      "google/gemma-3-270m-it"
  )

  model = transformers.AutoModelForSequenceClassification.from_pretrained(
      model_name, **model_kwargs
  )

  # Prepare model for QLoRA training
  if use_qlora:
    model = peft.prepare_model_for_kbit_training(model)

  # 3. Apply LoRA for Sequence Classification
  logger.info("Configuring LoRA adapters...")
  peft_config = peft.LoraConfig(
      r=lora_r,
      lora_alpha=lora_alpha,
      lora_dropout=0.05,
      bias="none",
      task_type="SEQ_CLS",  # Sequence Classification task type
      # no target_modules — PEFT's Gemma 4 defaults scope to the LM layers
      modules_to_save=["score"],
  )
  model = peft.get_peft_model(model, peft_config)

  # 4. Load & preprocess dataset
  logger.info("Loading dataset from %s...", dataset_path)
  raw_dataset = datasets.load_dataset(
      "json", data_files=dataset_path, split="train"
  )

  dataset = raw_dataset.train_test_split(test_size=test_size)

  # 5. Initialize RewardTrainer
  logger.info("Initializing RewardTrainer...")
  trainer = trl.RewardTrainer(
      model=model,
      processing_class=processor,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      args=trl.RewardConfig(
          output_dir=output_dir,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          num_train_epochs=epochs,
          learning_rate=learning_rate,
          bf16=supports_bf16(),
          fp16=not supports_bf16(),
          eval_strategy="epoch",
          save_strategy="epoch",
          max_length=max_length,
      ),
  )

  # 6. Train
  logger.info("Starting Reward Model training run...")
  trainer_stats = trainer.train()
  logger.info("Training completed. Stats: %s", trainer_stats)

  # 7. Save reward adapters
  logger.info("Saving Reward Model adapters to %s...", output_dir)
  model.save_pretrained(output_dir)
  processor.save_pretrained(output_dir)
  logger.info("Done!")


def main() -> None:
  """Parses CLI arguments and runs reward model training."""
  parser = argparse.ArgumentParser(description="Train a Gemma Reward Model")
  parser.add_argument(
      "--model",
      type=str,
      default="google/gemma-3-270m",
      help="Model repo or local path",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      required=True,
      help="Path to JSON/JSONL pairwise preference dataset",
  )
  parser.add_argument(
      "--test-size", type=float, default=0.2, help="dataset test split size"
  )
  parser.add_argument(
      "--output", type=str, default="./rm_output", help="Output directory"
  )
  parser.add_argument(
      "--max-len", type=int, default=2048, help="Max sequence length"
  )
  parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
  parser.add_argument(
      "--lora-alpha", type=int, default=32, help="LoRA alpha scaling"
  )
  parser.add_argument(
      "--batch-size", type=int, default=2, help="Batch size per GPU"
  )
  parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
  parser.add_argument(
      "--lr",
      type=float,
      default=1e-3,
      help="Learning rate (typically much higher than SFT, e.g., 1e-3)",
  )
  parser.add_argument(
      "--force-no-qlora", action="store_true", help="Disable QLoRA"
  )

  args = parser.parse_args()

  use_qlora = not args.force_no_qlora

  train_reward_model(
      model_name=args.model,
      dataset_path=args.dataset,
      test_size=args.test_size,
      output_dir=args.output,
      max_length=args.max_len,
      lora_r=args.lora_r,
      lora_alpha=args.lora_alpha,
      batch_size=args.batch_size,
      epochs=args.epochs,
      learning_rate=args.lr,
      use_qlora=use_qlora,
  )


if __name__ == "__main__":
  main()
