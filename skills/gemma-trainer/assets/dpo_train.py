#!/usr/bin/env python3
"""
Gemma Direct Preference Optimization (DPO) Script
Aligns fine-tuned Gemma models with pairwise preference datasets.
Uses Unsloth (for maximum single-GPU VRAM optimization) or standard TRL DPOTrainer.
"""

import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gemma-dpo")

# Try to import Unsloth
try:
    from unsloth import FastModel
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer() # Patches TRL's DPOTrainer internally for speedup
    HAS_UNSLOTH = True
    logger.info("Unsloth library detected! Fast DPOTrainer patched.")
except ImportError:
    HAS_UNSLOTH = False
    logger.info("Unsloth not found. Falling back to standard HF PEFT + TRL DPOTrainer.")

from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def format_dpo_example(example):
    """Return TRL's explicit conversational preference shape."""
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "chosen": [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


def load_model_and_processor(
    base_model_name: str,
    adapter_path: str,
    processor_name: str,
    max_length: int,
    use_unsloth: bool,
    use_qlora: bool,
):
    """Loads model and processor, configuring LoRA using either Unsloth or standard HF PEFT."""
    if use_unsloth:
        if processor_name:
            raise ValueError("--processor currently requires --force-hf.")
        if not HAS_UNSLOTH:
            raise RuntimeError("Unsloth is requested but not installed.")

        logger.info(f"Loading SFT adapter with Unsloth from {adapter_path}...")
        # Unsloth handles loading the base model automatically when given an adapter path
        model, processor = FastModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=max_length,
            dtype=None,  # Auto-detection
            load_in_4bit=use_qlora,  # Enable QLoRA
            full_finetuning=False,
            use_gradient_checkpointing="unsloth",  # Saves immense VRAM
        )

        from unsloth.chat_templates import get_chat_template
        processor = get_chat_template(processor, chat_template="gemma-4")

    else:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from peft import PeftModel

        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model_kwargs = dict(
            dtype=torch_dtype, # What torch dtype to use
            device_map="auto", # Let torch decide how to load the model
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

        logger.info(f"Loading base model {base_model_name} with HF transformers...")
        processor = AutoProcessor.from_pretrained(processor_name or adapter_path)
        model = AutoModelForMultimodalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )

        logger.info(f"Mounting trainable SFT adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
            adapter_name="default"
        )

    return model, processor

def train_dpo(
    base_model_name: str,
    adapter_path: str,
    processor_name: str,
    dataset_path: str,
    test_size: float,
    output_dir: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    use_unsloth: bool,
    use_qlora: bool,
):
    """Unified training runner for DPO using DPOConfig."""
    from transformers import set_seed
    set_seed(seed)
    model, processor = load_model_and_processor(
        base_model_name=base_model_name,
        adapter_path=adapter_path,
        processor_name=processor_name,
        max_length=max_length,
        use_unsloth=use_unsloth,
        use_qlora=use_qlora,
    )

    logger.info(f"Loading dataset from {dataset_path}...")
    raw_dataset = load_dataset("json", data_files=dataset_path, split="train")

    logger.info("Formatting dataset as explicit conversational preferences...")
    dataset = raw_dataset.map(format_dpo_example).train_test_split(
        test_size=test_size, seed=seed)

    logger.info("Initializing DPOTrainer...")
    # Standard PEFT DPOTrainer can also set ref_model=None to save memory,
    # computing the reference model outputs by disabling active LoRA adapter on-the-fly.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            save_strategy="epoch",
            eval_strategy="epoch",
            max_length=max_length,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=seed,
            data_seed=seed,
        ),
    )

    logger.info(f"Starting {'Unsloth' if use_unsloth else 'standard HF'} DPO training run...")
    trainer_stats = trainer.train()
    logger.info(f"Training completed. Stats: {trainer_stats}")

    # Save model/adapters
    logger.info(f"Saving preference-aligned adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align Gemma models using Direct Preference Optimization (DPO)")

    # Path configuration
    parser.add_argument("--base-model", type=str, default="google/gemma-4-E2B", help="The original base model name or HF repo")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the existing SFT PEFT adapter directory")
    parser.add_argument("--processor", type=str, default=None, help="HF processor repo/path (defaults to the SFT adapter directory; requires --force-hf)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSON/JSONL preference dataset file")
    parser.add_argument("--output", type=str, default="./dpo_output", help="Output directory")

    # Training configuration
    parser.add_argument("--max-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--test-size", type=float, default=0.2, help="dataset test split size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU (set to 1 to reduce OOM)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs (usually 1 is sufficient)")
    parser.add_argument("--lr", type=float, default=1e-6, help="DPO learning rate (typically much smaller than SFT, e.g., 1e-6)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset split and training")

    # Execution options
    parser.add_argument("--force-hf", action="store_true", help="Force HF standard training even if Unsloth is installed")
    parser.add_argument("--force-no-qlora", action="store_true", help="Disable QLoRA")

    args = parser.parse_args()

    use_unsloth = HAS_UNSLOTH and not args.force_hf
    use_qlora = not args.force_no_qlora

    train_dpo(
        base_model_name=args.base_model,
        adapter_path=args.adapter,
        processor_name=args.processor,
        dataset_path=args.dataset,
        test_size=args.test_size,
        output_dir=args.output,
        max_length=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        use_unsloth=use_unsloth,
        use_qlora=use_qlora,
    )
