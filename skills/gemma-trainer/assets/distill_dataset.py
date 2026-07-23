#!/usr/bin/env python3
"""
Gemma Dataset Distillation & Synthesis Script
This script allows users to generate high-quality fine-tuning datasets locally using 
a larger "teacher" model (e.g., Gemma 4 31B, 26B A4B) to train a smaller 
"student" model (e.g., Gemma 4 E2B, Gemma 4 E4B).

It supports:
1. Response Distillation: Populates responses for a list of raw input prompts.
2. Self-Instruct Synthesis: Generates new instruction-response pairs from seed topics.
3. Multi-backend: Supports Hugging Face (with/without 4-bit quantization) or local Ollama instances.
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gemma-distill")


def supports_bf16() -> bool:
    """Checks bf16 support on the active accelerator (CUDA or Apple Silicon MPS).

    torch.cuda.is_bf16_supported() only checks CUDA, so on MPS-only machines
    (e.g. Apple Silicon Macs) it always reports False and generation silently
    falls back to fp16.
    """
    import torch
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    if torch.backends.mps.is_available():
        return torch.backends.mps.is_macos13_or_newer()
    return False


def load_seed_prompts(input_path: str) -> List[str]:
    """Loads seed prompts from a TXT, JSON, or JSONL file."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
        
    prompts = []
    _, ext = os.path.splitext(input_path.lower())
    
    if ext == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    elif ext in [".json", ".jsonl"]:
        with open(input_path, "r", encoding="utf-8") as f:
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            # Try to extract prompt/instruction/text field
                            p = obj.get("prompt") or obj.get("instruction") or obj.get("text")
                            if p:
                                prompts.append(str(p))
                    except json.JSONDecodeError:
                        continue
            else:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for obj in data:
                            if isinstance(obj, dict):
                                p = obj.get("prompt") or obj.get("instruction") or obj.get("text")
                                if p:
                                    prompts.append(str(p))
                            elif isinstance(obj, str):
                                prompts.append(obj)
                    elif isinstance(data, dict):
                        p = data.get("prompt") or data.get("instruction") or data.get("text")
                        if p:
                            prompts.append(str(p))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON file: {e}")
                    sys.exit(1)
    else:
        logger.error("Unsupported seed prompt file format. Please use .txt, .json, or .jsonl")
        sys.exit(1)
        
    logger.info(f"Loaded {len(prompts)} seed prompts/topics from {input_path}")
    return prompts

def init_hf_pipeline(model_id: str, use_qlora):
    """Initializes Hugging Face model and processor."""
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM, pipeline, GenerationConfig
    except ImportError:
        logger.error(
            "Hugging Face libraries are not installed. Please run:\n"
            "pip install transformers torch torchvision librosa pillow accelerate"
        )
        sys.exit(1)

    logger.info(f"Loading HF teacher model '{model_id}'...")
    
    torch_dtype = torch.bfloat16 if supports_bf16() else torch.float16

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
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    gen_config = GenerationConfig.from_pretrained(model_id)
    
    gen_pipeline = pipeline(
        "any-to-any",
        model=model,
        processor=processor,
    )
    return gen_pipeline, gen_config

def generate_response_hf(
    pipeline_obj,
    gen_kwargs,
    prompt: str, 
    system_prompt: str, 
) -> str:
    """Generates a response using Hugging Face pipeline with chat formatting."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    outputs = pipeline_obj(
        messages,
        return_full_text=False,
        generate_kwargs=gen_kwargs,
    )
    
    return outputs[0]["generated_text"].strip().removesuffix("<turn|>")

def query_ollama_api(url: str, model: str, messages: List[Dict[str, str]], temp: float, max_tokens: int) -> str:
    """Queries local Ollama endpoint via HTTP request."""
    import urllib.request
    import urllib.error
    
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temp,
            "num_predict": max_tokens
        },
        "think": False,
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["message"]["content"].strip()
    except urllib.error.URLError as e:
        logger.error(f"Failed to communicate with Ollama API: {e}. Is Ollama running?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error calling Ollama: {e}")
        sys.exit(1)

def run_response_distillation(args) -> List[Dict[str, Any]]:
    """Runs response population mode."""
    prompts = load_seed_prompts(args.input)
    dataset = []
    
    if args.use_ollama:
        logger.info(f"Using Ollama API ({args.ollama_url}) with model '{args.ollama_model}' for generation...")
        for i, p in enumerate(prompts, 1):
            logger.info(f"[{i}/{len(prompts)}] Generating response for prompt: {p[:60]}...")
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            messages.append({"role": "user", "content": p})
            
            response = query_ollama_api(
                url=args.ollama_url,
                model=args.ollama_model,
                messages=messages,
                temp=args.temperature,
                max_tokens=args.max_new_tokens
            )
            
            # Format to the standard conversational SFT messages format
            turn = {
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "model", "content": response}
                ]
            }
            if args.system_prompt:
                turn["messages"].insert(0, {"role": "system", "content": args.system_prompt})
            dataset.append(turn)
    else:
        from transformers import TextStreamer
        pipeline_obj, config_obj = init_hf_pipeline(args.model)
        for i, p in enumerate(prompts, 1):
            logger.info(f"[{i}/{len(prompts)}] Generating response for prompt: {p[:60]}...")
            config_obj.max_new_tokens=args.max_new_tokens
            config_obj.temperatur=args.temperature
            gen_kwargs = dict(generation_config=config_obj, streamer=TextStreamer(pipeline_obj.tokenizer))
            response = generate_response_hf(
                pipeline_obj=pipeline_obj,
                gen_kwargs=gen_kwargs,
                prompt=p,
                system_prompt=args.system_prompt,
            )
            
            turn = {
                "messages": [
                    {"role": "user", "content": p},
                    {"role": "model", "content": response}
                ]
            }
            if args.system_prompt:
                turn["messages"].insert(0, {"role": "system", "content": args.system_prompt})
            dataset.append(turn)
            
    return dataset

def run_self_instruct_synthesis(args) -> List[Dict[str, Any]]:
    """Runs self-instruct style prompt and response generation."""
    seed_prompts = []
    if args.input:
        seed_prompts = load_seed_prompts(args.input)
    else:
        # Default fallback seed categories if no file is provided
        seed_prompts = [
            "Logical reasoning, mathematical puzzle solving, and algorithmic programming.",
            "Technical writing, clear explanation of complex system design, and API documentation.",
            "Creative writing, copywriting, roleplay, and stylistic modification.",
            "Language translation, text simplification, summarizing, and structured JSON parsing."
        ]
        logger.info(f"No seed input provided. Using {len(seed_prompts)} default general seed topics.")
        
    dataset = []
    
    # Synthesize prompt instructs the teacher model to brainstorm instructions and responses
    system_instruction = (
        "You are an advanced synthetic dataset generator. Your goal is to produce high-quality, "
        "extremely detailed, and varied instruction-following pairs for training smaller language models."
    )
    
    # We will generate in batches to handle diverse outputs and maintain reliability
    samples_per_batch = 5
    num_batches = (args.num_samples + samples_per_batch - 1) // samples_per_batch
    
    logger.info(f"Generating {args.num_samples} synthetic samples in {num_batches} batches...")
    
    prompt_template = (
        "We are building a fine-tuning dataset. Here are some seed categories and sample topics:\n"
        f"{json.dumps(seed_prompts, indent=2)}\n\n"
        f"Generate exactly {samples_per_batch} diverse, complex, and creative instruction-following samples. "
        "Each sample MUST have a distinct topic and challenge (e.g., code debugging, creative rewriting, "
        "step-by-step logic, or structured multi-step thinking).\n\n"
        "You MUST return the output as a valid JSON array of objects. Do not wrap it in markdown code blocks. "
        "Do not write any preamble or conversational explanation before/after the JSON. Just output raw valid JSON.\n"
        "Format schema:\n"
        "[\n"
        "  {\n"
        "    \"messages\": [\n"
        "      {\"role\": \"user\", \"content\": \"<detailed instruction/prompt>\"},\n"
        "      {\"role\": \"model\", \"content\": \"<high-quality comprehensive expert response>\"}\n"
        "    ]\n"
        "  }\n"
        "]"
    )
    
    if args.use_ollama:
        logger.info(f"Using Ollama API ({args.ollama_url}) with model '{args.ollama_model}' for synthesis...")
        for b in range(1, num_batches + 1):
            logger.info(f"Processing batch {b}/{num_batches}...")
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_template}
            ]
            raw_response = query_ollama_api(
                url=args.ollama_url,
                model=args.ollama_model,
                messages=messages,
                temp=args.temperature,
                max_tokens=args.max_new_tokens
            )
            print(raw_response)
            
            # Clean up potential markdown wrappers
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            try:
                batch_data = json.loads(cleaned)
                if isinstance(batch_data, list):
                    dataset.extend(batch_data)
                    logger.info(f"Added {len(batch_data)} synthetic samples from batch {b}.")
                else:
                    logger.warning(f"Batch {b} did not return a list. Response content: {cleaned[:100]}")
            except Exception as e:
                logger.error(f"Failed to parse JSON from batch {b}: {e}. Retrying or skipping.")
                continue
    else:
        from transformers import TextStreamer
        pipeline_obj, config_obj = init_hf_pipeline(args.model, not args.force_no_qlora)
        for b in range(1, num_batches + 1):
            config_obj.max_new_tokens=args.max_new_tokens
            config_obj.temperatur=args.temperature
            gen_kwargs = dict(generation_config=config_obj, streamer=TextStreamer(pipeline_obj.tokenizer))

            logger.info(f"Processing batch {b}/{num_batches}...")
            raw_response = generate_response_hf(
                pipeline_obj=pipeline_obj,
                gen_kwargs=gen_kwargs,
                prompt=prompt_template,
                system_prompt=system_instruction,
            )
            
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            try:
                batch_data = json.loads(cleaned)
                if isinstance(batch_data, list):
                    dataset.extend(batch_data)
                    logger.info(f"Added {len(batch_data)} synthetic samples from batch {b}.")
                else:
                    logger.warning(f"Batch {b} did not return a list. Response content: {cleaned[:100]}")
            except Exception as e:
                logger.error(f"Failed to parse JSON from batch {b}: {e}.")
                continue
                
    # Truncate to exact user requested samples
    return dataset[:args.num_samples]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Dataset Distillation & Synthesis Tool")
    parser.add_argument("--mode", type=str, choices=["response", "synthesize"], required=True, help="Distillation mode: 'response' (populates answers for prompts) or 'synthesize' (self-instruct dataset synthesis)")
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it", help="HF repository ID of the teacher model (used for Hugging Face backend)")
    parser.add_argument("--input", type=str, help="Path to TXT, JSON, or JSONL file containing seed prompts or categories")
    parser.add_argument("--output", type=str, default="./distilled_dataset.json", help="Path to write the generated fine-tuning dataset JSON file")
    parser.add_argument("--system-prompt", type=str, default="You are an helpful, expert AI assistant. Provide extremely accurate, polished, and structured answers.", help="System instruction given to the teacher model during response generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (higher values increase creativity/variance)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generation response length (tokens) per prompt")
    parser.add_argument("--num-samples", type=int, default=50, help="For 'synthesize' mode, total number of synthetic training samples to generate")
    parser.add_argument("--force-no-qlora",action="store_true",help="Disable QLoRA")
    
    # Ollama Options
    parser.add_argument("--use-ollama", action="store_true", help="Use a locally running Ollama instance as the teacher backend instead of Hugging Face")
    parser.add_argument("--ollama-model", type=str, default="gemma4:31b", help="Ollama model name to query (e.g., 'gemma4:31b', 'gemma4:26b')")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="URL of the local Ollama service endpoint")

    args = parser.parse_args()

    if args.mode == "response" and not args.input:
        parser.error("--input is required when --mode is set to 'response'")

    logger.info("====================================================")
    logger.info("Starting Gemma Dataset Distillation / Generation")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output target: {args.output}")
    logger.info("====================================================")

    if args.mode == "response":
        result_dataset = run_response_distillation(args)
    else:
        result_dataset = run_self_instruct_synthesis(args)

    if result_dataset:
        logger.info(f"Saving final dataset containing {len(result_dataset)} items...")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"SUCCESS: Distilled dataset successfully saved to '{args.output}'!")
    else:
        logger.error("Failed to generate any valid samples. Output file not created.")
        sys.exit(1)
