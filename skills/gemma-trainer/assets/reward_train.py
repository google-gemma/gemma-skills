#!/usr/bin/env python3
"""
Gemma Reward Modeling Script
Trains a classification-style reward head on top of a Gemma base model.
Uses Hugging Face PEFT + TRL RewardTrainer.

NOTE: AutoModelForSequenceClassification doesn't support Gemma 4 yet.
      https://github.com/huggingface/transformers/issues/45373
"""

import torch
import logging

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import RewardTrainer, RewardConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gemma-reward-model")

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
):
    """Fine-tunes a Gemma sequence classification model as a Reward Model."""

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs = dict(
        dtype=torch_dtype, # What torch dtype to use
        device_map="auto", # Let torch decide how to load the model
        num_labels=1, # sequence classification (1 label)
    )

    if use_qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
        )

    # Load model & processor
    logger.info(f"Loading {model_name} with classification head (num_labels=1)...")
    processor = AutoProcessor.from_pretrained("google/gemma-3-270m-it")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Prepare model for QLoRA training
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # 3. Apply LoRA for Sequence Classification
    logger.info("Configuring LoRA adapters...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS", # Sequence Classification task type
        # no target_modules — PEFT's Gemma 4 defaults scope to the LM layers
        modules_to_save=["score"]
    )
    model = get_peft_model(model, peft_config)
    
    # 4. Load & preprocess dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    dataset = raw_dataset.train_test_split(test_size=test_size)

    # 5. Initialize RewardTrainer
    logger.info("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=RewardConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            eval_strategy="epoch",
            save_strategy="epoch",
            max_length=max_length,
        ),
    )

    # 6. Train
    logger.info("Starting Reward Model training run...")
    trainer_stats = trainer.train()
    logger.info(f"Training completed. Stats: {trainer_stats}")

    # 7. Save reward adapters
    logger.info(f"Saving Reward Model adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a Gemma Reward Model")
    parser.add_argument("--model", type=str, default="google/gemma-3-270m", help="Model repo or local path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSON/JSONL pairwise preference dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="dataset test split size")
    parser.add_argument("--output", type=str, default="./rm_output", help="Output directory")
    parser.add_argument("--max-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (typically much higher than SFT, e.g., 1e-3)")
    parser.add_argument("--force-no-qlora", action="store_true", help="Disable QLoRA")

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
