#!/usr/bin/env python3
"""
Gemma Supervised Fine-Tuning (SFT) Script
This script fine-tunes Gemma models on local environments using Unsloth (for maximum 
efficiency on single GPUs) or standard Hugging Face PEFT + TRL (fallback).
"""

import torch
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gemma-sft")

# Attempt to import Unsloth for high-performance single-GPU training
try:
    from unsloth import FastVisionModel
    HAS_UNSLOTH = True
    logger.info("Unsloth library detected! Will use optimized single-GPU training.")
except ImportError:
    HAS_UNSLOTH = False
    logger.info("Unsloth not found. Falling back to standard Hugging Face PEFT + TRL.")

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from training_utils import mask_assistant_tokens, processor_kwargs

def load_model_and_processor(
    model_name: str,
    max_length: int,
    lora_r: int,
    lora_alpha: int,
    use_unsloth: bool,
    use_qlora: bool,
):
    """Loads model and processor, configuring LoRA using either Unsloth or standard HF PEFT."""
    # Determine reference IT model for processor fallback if base PT model is used
    processor_model_id = model_name if "-it" in model_name else f"{model_name}-it"
    fallback_model_id = "google/gemma-4-E2B-it"

    if use_unsloth:
        if not HAS_UNSLOTH:
            raise RuntimeError("Unsloth is requested but not installed.")

        logger.info(f"Loading {model_name} with Unsloth...")
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,  # Auto-detection
            load_in_4bit=use_qlora,  # Enable QLoRA
            full_finetuning=False,
            use_gradient_checkpointing="unsloth",  # Saves immense VRAM
        )

        from unsloth.chat_templates import get_chat_template
        processor = get_chat_template(processor, chat_template="gemma-4")

        logger.info("Configuring LoRA adapters...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = False, # Start with language-side adaptation
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers

            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0,  # Unsloth optimized setting
            bias="none",
            # no target_modules — PEFT's Gemma 4 defaults scope to the LM layers
        )
    else:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

            # === FIXED AUDIO DATA TYPE PATCH ===
            _original_masked_scatter = torch.Tensor.masked_scatter

            def _patched_masked_scatter(self, mask, source):
                if self.dtype != source.dtype:
                    source = source.to(self.dtype)
                return _original_masked_scatter(self, mask, source)

            torch.Tensor.masked_scatter = _patched_masked_scatter
            # ===================================

        # Load model & processor
        try:
            logger.info(f"Loading processor config from {processor_model_id}...")
            processor = AutoProcessor.from_pretrained(processor_model_id)
        except OSError:
            logger.warning(f"Failed to load {processor_model_id}. Falling back to {fallback_model_id}.")
            processor = AutoProcessor.from_pretrained(fallback_model_id)
        
        logger.info(f"Loading base model weights from {model_name}...")        
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Prepare model for QLoRA training
        if use_qlora:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # no target_modules — PEFT's Gemma 4 defaults scope to the LM layers
        )
        model = get_peft_model(model, peft_config)

    return model, processor

def get_collate_fn(processor, max_length):
    """Creates a custom data collator that masks user prompts in loss computation."""
    def collate_fn(examples):
        texts = []
        audios = []
        images = []

        # Check if ANY example in the entire batch actually contains audio
        batch_has_audio = False
        for example in examples:
            for message in example.get("messages", []):
                content = message.get("content", [])
                if isinstance(content, list):
                    if any(isinstance(b, dict) and b.get("type") == "audio" for b in content):
                        batch_has_audio = True
                        break

        if batch_has_audio:
            try:
                import librosa
            except ImportError:
                raise ImportError(
                    "Your dataset contains audio blocks, but `librosa` is not installed. "
                    "Please install librosa using `pip install librosa` to train on audio data."
                )

        for example in examples:
            messages = example.get("messages", [])

            full_text = processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            texts.append(full_text.strip())

            # images can be nested, not for audio
            example_images = []
            example_audio = np.zeros(1, dtype=np.float32) if batch_has_audio else None

            for message in messages:
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue

                        if block.get("type") == "audio":
                            audio_path = block.get("audio") or block.get("url")

                            try:
                                audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
                                if audio_array.ndim > 1:
                                    audio_array = audio_array.squeeze()

                                example_audio = audio_array
                            except Exception as e:
                                print(f"Error loading audio file {audio_path}: {e}")
                                example_audio = np.zeros(1, dtype=np.float32)

                        elif block.get("type") == "image":
                            image_path = block.get("image") or block.get("url")
                            if image_path:
                                example_images.append(image_path)

            images.append(example_images)
            if batch_has_audio:
                audios.append(example_audio)

        batch_kwargs = processor_kwargs(texts, max_length)

        if batch_has_audio:
            batch_kwargs["audio"] = audios
        if any(len(img_list) > 0 for img_list in images):
            batch_kwargs["images"] = images

        # Tokenize the texts
        batch = processor(**batch_kwargs)

        target_phrase = "<|turn>model\n"
        target_tokens = processor.tokenizer.encode(target_phrase, add_special_tokens=False)
        turn_end_tokens = processor.tokenizer.encode("<turn|>", add_special_tokens=False)
        labels = batch["input_ids"].clone()

        for i in range(labels.size(0)):
            row_tokens = batch["input_ids"][i].tolist()
            masked = mask_assistant_tokens(
                row_tokens,
                assistant_start=target_tokens,
                turn_end=turn_end_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            labels[i] = torch.tensor(
                masked, dtype=labels.dtype, device=labels.device
            )

        batch["labels"] = labels

        return batch

    return collate_fn


def train(
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
    use_unsloth: bool,
    use_qlora: bool,
):
    """Unified training runner for SFT using SFTConfig."""
    model, processor = load_model_and_processor(
        model_name=model_name,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_unsloth=use_unsloth,
        use_qlora=use_qlora,
    )

    # Load and format dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train").train_test_split(test_size=test_size)

    if use_unsloth:
        from unsloth.trainer import UnslothVisionDataCollator

        collate_fn = UnslothVisionDataCollator(
            model,
            processor,
            max_seq_length=max_length,
            train_on_responses_only=True,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
            completion_only_loss=True,
        )
    else:
        collate_fn = get_collate_fn(processor, max_length)
    total_steps = (len(dataset['train']) // batch_size) * epochs

    # Configure trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=int(total_steps * 0.03), # 3% warmup ratio
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            eval_strategy="epoch",
            save_strategy="epoch",
            remove_unused_columns=False,                   # important for collator
            dataset_kwargs={"skip_prepare_dataset": True}, # important for collator
            max_length=max_length,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", # track loss, not accuracy
            greater_is_better=False,           # Lower loss is better
        ),
    )

    # Start Training
    logger.info(f"Starting {'Unsloth' if use_unsloth else 'standard HF'} training run...")
    trainer_stats = trainer.train()
    logger.info(f"Training completed. Stats: {trainer_stats}")

    # Save model/adapters
    logger.info(f"Saving fine-tuned adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune Gemma models using SFT QLoRA")
    parser.add_argument("--model", type=str, default="google/gemma-4-E2B", help="Model repo or local path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSON/JSONL dataset file")
    parser.add_argument("--test-size", type=float, default=0.2, help="dataset test split size")
    parser.add_argument("--output", type=str, default="./sft_output", help="Output directory")
    parser.add_argument("--max-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--force-hf", action="store_true", help="Force HF standard training even if Unsloth is installed")
    parser.add_argument("--force-no-qlora", action="store_true", help="Disable QLoRA")

    args = parser.parse_args()

    use_unsloth = HAS_UNSLOTH and not args.force_hf
    use_qlora = not args.force_no_qlora

    train(
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
        use_unsloth=use_unsloth,
        use_qlora=use_qlora,
    )
