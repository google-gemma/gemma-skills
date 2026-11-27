---
name: gemma-trainer
description: >-
  Guides local and multi-GPU post-training, fine-tuning, alignment, scaling, and
  export for Gemma 4 models across Unsloth and Hugging Face TRL/PEFT. Use when
  fine-tuning or adapting Gemma models with SFT, (Q)LoRA, Full Fine-Tuning,
  26B-A4B MoE, Continued Pre-Training, Custom Chat Templates, Unused Tokens,
  Vision/Audio multimodal SFT, DPO, GRPO verifiable rewards, Reward Modeling,
  PyTorch FSDP2, TorchAO QAT, dataset validation/distillation, or vLLM, GGUF,
  and LiteRT-LM export. Don't use for inference-only application building or
  general model selection without training (use gemma-dev).
---

# Gemma Training and Fine-Tuning Skill

## 1. Core Principles: Local & Multi-GPU Setup

When configuring a Gemma 4 training run, match the stack to the user's hardware,
checkpoint architecture, and modality:

*   **Prioritize Unsloth on a Single GPU**: For local single-GPU LoRA or QLoRA
    training, recommend **Unsloth** (`FastVisionModel` / `FastModel`). It
    supports Gemma 4 natively (including 4-bit `26B-A4B` MoE), uses up to 60–70%
    less VRAM, and trains up to 2x faster than standard PEFT.
*   **Use Hugging Face TRL + PEFT for Multi-GPU or Custom Pipelines**: For
    multi-GPU environments (using DDP or **PyTorch FSDP2** via
    `accelerate launch`), full fine-tuning, custom `trainable_token_indices`, or
    TorchAO QAT, use Hugging Face **TRL** (`SFTTrainer`, `DPOTrainer`,
    `GRPOTrainer`, `RewardTrainer`) with **PEFT**.
*   **Size Quantization & Optimizer to the Checkpoint**:
    *   **Dense text & vision (`E2B`, `E4B`, `12B`, `31B`)**: Use **QLoRA (4-bit
        NF4 quantization)** via `BitsAndBytesConfig` or Unsloth to fit consumer
        16–24 GB VRAM.
    *   **MoE (`26B-A4B`)**: Routed 128 experts are fused 3D `nn.Parameter`
        tensors (`45.7 GB` in `bfloat16`) that `BitsAndBytesConfig` skips. On a
        24–40 GB single GPU, use Unsloth `FastVisionModel(load_in_4bit=True)`;
        in pure HF PEFT on 80 GB / FSDP2, load in `bfloat16` and pass
        `target_parameters=["experts.gate_up_proj", "experts.down_proj"]`.
    *   **Audio (`E2B`, `E4B`, `12B` only)**: `26B-A4B` and `31B` have no audio
        encoder! Always train audio LoRA in **`bfloat16`** (uniform 4-bit NF4
        degrades audio projections).
    *   **Full Fine-Tuning on `E2B` / `E4B` (Per-Layer Embeddings)**: Never use
        `bitsandbytes` 8-bit AdamW (`paged_adamw_8bit`), which leaves
        `nn.Embedding` / `embed_tokens_per_layer` optimizer states in `fp32`
        (+22 GB VRAM OOM). Always set **`optim="adamw_torch_8bit"`**
        (`torchao`).
*   **Manage Context Window & Max Length**: Although Gemma 4 supports 128K–256K
    inference context, limit local training `max_length` / `max_seq_length` to
    **2048–8192 tokens** locally to prevent Out-Of-Memory (OOM) errors.
*   **Evaluate Before & After + Enable `eval_on_start=True`**: Whenever an
    `eval_dataset` is provided, pass `eval_on_start=True` so Epoch 0 baseline
    validation loss is logged before step 1, and run held-out task evaluation
    both before and after `trainer.train()`.

--------------------------------------------------------------------------------

## 2. Choosing the Right Training Method

Route the user's goal to the right post-training stage and checkpoint type
(default to instruction-tuned `-it` checkpoints; use base `pt` checkpoints only
for Continued Pre-Training or custom chat templates):

*   **Continued Pre-Training (CPT)**: Adapting a base (`pt`) checkpoint (e.g.
    `google/gemma-4-E2B`) to raw domain corpora or underrepresented languages.
    Use sequence packing (`packing=True`, `eval_packing=False`), `lr=5e-5`
    (`1e-5` on unfrozen embeddings), and mix a **10–20% general replay buffer**
    to prevent catastrophic forgetting.
*   **Custom Chat Templates**: Teaching a base (`pt`) model custom turn
    delimiters (e.g. ChatML `<|im_start|>` / `<|im_end|>`). Mark assistant spans
    in Jinja2 with `{% generation %}...{% endgeneration %}` and set
    `assistant_only_loss=True`.
*   **Supervised Fine-Tuning (SFT — LoRA, QLoRA, Full FT, MoE)**: Teaching
    specialized tasks, domain schemas, personas, or multimodal extraction on
    prompt/completion or conversational pairs.
*   **Unused Control / Routing Tokens**: Adding single-token domain routing or
    source tags without resizing the `262,144` vocabulary. Register built-in
    `<unused0>`–`<unused6226>` slots via
    `AddedToken(slot, special=False, normalized=False)`, initialize embeddings
    (including `embed_tokens_per_layer` when present) from subword means, and
    train via PEFT `trainable_token_indices`.
*   **Direct Preference Optimization (DPO)**: Aligning response tone,
    conciseness, style, or safety on pairwise `(prompt, chosen, rejected)` data
    after SFT without a separate reward model.
*   **Reinforcement Learning with Verifiable Rewards (GRPO)**: Teaching concise
    reasoning budgets (`<|channel>thought...<channel|>`) or strict verifiable
    output contracts (math, code execution, SQL, parsers) using TRL
    `GRPOTrainer` (`num_generations=4..8`, `beta=0.0`,
    `chat_template_kwargs={"enable_thinking": True}`).
*   **Reward Modeling (RM)**: Training a scalar scoring head
    (`AutoModelForSequenceClassification` with `num_labels=1`) on binary
    preference pairs to score response quality.

--------------------------------------------------------------------------------

## 3. Dataset Preparation & Validation

Formatting issues are the #1 cause of poor training runs. Validate local files
using `[assets/dataset_prep.py]`.

### Gemma 4 Chat & Thinking Format

Ensure prompts match Gemma 4's official turn and thinking delimiters via
`tokenizer.apply_chat_template` / `processor.apply_chat_template`:

```text
<|turn>system
Your instruction here<turn|>
<|turn>user
Your query here<turn|>
<|turn>model
<|channel>thought
Optional reasoning trace<channel|>Your response here<turn|>
```

### Format Specifications

*   **SFT (Supervised Fine-Tuning)**: Supports two structures:

    *   **Conversational (`messages`)**: List of role/content turns (pass
        `completion_only_loss=True` in `SFTConfig` to mask user/system prompts):

        ```json
        {
          "messages": [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why did the computer go to the doctor? It had a virus!"}
          ]
        }
        ```
    *   **Prompt-Completion (`prompt` and `completion`)**: `SFTTrainer` masks
        the `prompt` automatically (`completion_only_loss=True` by default):

        ```json
        {
          "prompt": [
            {"role": "system", "content": "You are an expert SQL assistant."},
            {"role": "user", "content": "Write a query to count active users."}
          ],
          "completion": [
            {"role": "assistant", "content": "SELECT COUNT(*) FROM users WHERE active = 1;"}
          ]
        }
        ```
*   **DPO & Reward Modeling**: Pairwise samples with `prompt`, `chosen`, and
    `rejected`:

    ```json
    {
      "prompt": "Write a python function to compute factorial.",
      "chosen": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
      "rejected": "factorial is computed using recursion or loops. Just import math."
    }
    ```

### Dataset Distillation & Synthesis (Teacher-Student)

To generate synthetic training data for compact models (`E2B` / `E4B`) from a
larger teacher (`31B` or `26B-A4B`), use `[assets/distill_dataset.py]`.

--------------------------------------------------------------------------------

## 4. Fine-Tuning Workflows & Key Guardrails

### Supervised Fine-Tuning (SFT) & MoE

Use `[assets/sft_train.py]` for quick local single-GPU SFT, or fetch the exact
canonical guide in Section 7 for specialized setups.

*   **LoRA Hyperparameters**:
    *   Rank (`r`): `16` or `32`; Alpha (`lora_alpha`): `32` or `64`
        (`lora_alpha = 2 * r`); Dropout (`lora_dropout`): `0.0` (Unsloth) or
        `0.05`.
    *   Learning Rate: `2e-4` for (Q)LoRA; `2e-5`–`5e-5` with cosine schedule +
        `10%` warmup for full fine-tuning (`optim="adamw_torch_8bit"`).
    *   **Exclude Frozen Multimodal Towers**: In HF PEFT `LoraConfig`, always
        pass `exclude_modules=r"model\.(vision_tower|audio_tower)\..*"` so PEFT
        skips custom `Gemma4ClippableLinear` layers inside frozen vision/audio
        encoders.
    *   **`26B-A4B` MoE LoRA**: Pass both `target_modules=["q_proj", "k_proj",
        "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` and
        `target_parameters=["experts.gate_up_proj", "experts.down_proj"]`, and
        freeze the router via
        `exclude_modules=r"(model\.(vision_tower|audio_tower)\..*|.*router.*)"`.

### Direct Preference Optimization (DPO), GRPO & Reward Modeling

*   **DPO (`[assets/dpo_train.py]`)**: Always perform SFT on your target
    instruction format **before** running DPO (running DPO directly on an
    unaligned base model degrades formatting). Start with `beta=0.1`
    (`0.1`–`0.5` controls reference adherence) and `ref_model=None` when using
    PEFT LoRA.
*   **GRPO (`GRPOTrainer`)**: Needs only a `prompt` column plus ground-truth
    columns for deterministic reward functions (e.g., concise closed thought
    trace + regex `Answer:` parser + exact answer match). Set `beta=0.0` and
    `num_generations=4..8`.
*   **Reward Modeling (`[assets/reward_train.py]`)**: Initialize with
    `AutoModelForSequenceClassification` (`num_labels=1`) to train a scalar
    quality head on `(prompt, chosen, rejected)` pairs.

--------------------------------------------------------------------------------

## 5. Multimodal Fine-Tuning (Vision & Audio)

Always place `image` or `audio` content items **before** `text` inside user
turns, and mask all prompt tokens plus `<|image|>` / `<|audio|>` soft tokens to
`-100` in your custom `SFTTrainer` data collator.

### Vision SFT

*   Supported checkpoints: `E2B`, `E4B`, `12B` (Unified), `26B-A4B`, `31B`.
*   Configure token budget via `processor.image_processor.max_soft_tokens`
    (`70`, `140`, `280`, `560`, or `1120`) before training and save the
    processor alongside the model:

    ```json
    {
      "messages": [
        {"role": "user", "content": [
            {"type": "image", "url": "path/to/image.png"},
            {"type": "text", "text": "Extract the receipt line items as JSON."}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "{\"total\": \"14.50\"}"}
        ]}
      ]
    }
    ```

### Audio SFT

*   Supported checkpoints: **`E2B`, `E4B`, or `12B` (Unified) only** (`26B-A4B`
    and `31B` are text-and-vision only).
*   Resample raw audio waveforms to **16kHz** mono (`soundfile` / `librosa`) and
    train with **`bfloat16` LoRA** (`use_reentrant=False`):

    ```json
    {
      "messages": [
        {"role": "user", "content": [
            {"type": "audio", "url": "path/to/audio.wav"},
            {"type": "text", "text": "Transcribe this speech."}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "This is an audio file of a bird chirping."}
        ]}
      ]
    }
    ```

--------------------------------------------------------------------------------

## 6. Scaling, Quantization-Aware Training & Export

*   **Multi-GPU Sharding (PyTorch FSDP2)**: Shard weights, gradients, and
    optimizer states across GPUs with Accelerate `FSDP` (`fsdp_version: 2`,
    `TRANSFORMER_BASED_WRAP`). For multi-GPU 4-bit QLoRA data parallelism, pin
    each worker with `device_map={"": PartialState().process_index}`.
*   **Post-Tuning Quantization-Aware Training (TorchAO QAT)**: If post-training
    4-bit rounding hurts accuracy after custom `bfloat16` fine-tuning, run a
    short recovery pass (500–1,500 steps, `lr=1e-5`) using TorchAO `QATConfig`
    (`Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4,
    weight_granularity=PerGroup(32))`) filtered to
    `model.language_model.layers.*` linear modules to simulate 4-bit rounding
    before converting to packed `int4`.
*   **vLLM (Cloud / GPU Serving)**: Serve `merged_model/` directly with
    `vllm serve ./merged_model --dtype bfloat16`, or attach multiple unmerged
    adapters dynamically via `--enable-lora --lora-modules ...`.
*   **GGUF Conversion (`llama.cpp` / Ollama)**:
    *   **Option 1 — Unsloth Native Export (Recommended)**: Use
        `model.save_pretrained_gguf(..., quantization_method="q4_k_m")` (see
        [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf.md)).
    *   **Option 2 — Manual `llama.cpp` Conversion**: Run
        `convert_hf_to_gguf.py ./merged_model --outtype f16` followed by
        `llama-quantize ... Q4_K_M`.
*   **On-Device Deployment with LiteRT-LM (`.litertlm`)**: For mobile, web, and
    IoT NPU/GPU acceleration on **`E2B` and `E4B`**, export with the LiteRT-LM
    CLI tool
    (`litert-torch export_hf --model=./merged_model --externalize_embedder`) so
    Per-Layer Embeddings stay separated (see
    [LiteRT-LM guide](https://gemma.ai/deployment/litert.md.txt)).

--------------------------------------------------------------------------------

## 7. Documentation Lookup (Live Canonical Code on `gemma.ai`)

Whenever generating or customizing training scripts beyond the bundled
`[assets/*.py]` templates, retrieve the live official guide so your code uses
the latest verified package versions, collators, and hyperparameters.

### When MCP is Installed (Preferred)

If the **`search_documentation`** tool (from the Google MCP server) is
available, query `search_documentation` first and trust its results as the
up-to-date source of truth.

### When MCP is NOT Installed (Fallback / URL Fetch)

Use `fetch_url` (or `curl`) to read the index at `https://gemma.ai/llms.txt` or
fetch the specific raw Markdown guide directly:

*   [Fine-tuning overview & hardware sizing](https://gemma.ai/tuning.md.txt)
*   [Best practices & hyperparameters](https://gemma.ai/tuning/best-practices.md.txt)
*   [Dataset formatting & prompt masking](https://gemma.ai/tuning/datasets-formatting.md.txt)
*   [Continued pre-training](https://gemma.ai/tuning/continued-pretraining.md.txt)
*   [Custom chat templates](https://gemma.ai/tuning/custom-chat-template.md.txt)
*   [(Q)LoRA fine-tuning](https://gemma.ai/tuning/qlora.md.txt)
*   [Full fine-tuning (`adamw_torch_8bit`)](https://gemma.ai/tuning/full-finetune.md.txt)
*   [MoE fine-tuning (`26B-A4B`)](https://gemma.ai/tuning/moe-finetune.md.txt)
*   [Unused tokens (`<unused*>`)](https://gemma.ai/tuning/unused-tokens.md.txt)
*   [Vision fine-tuning](https://gemma.ai/tuning/vision-qlora.md.txt)
*   [Audio fine-tuning](https://gemma.ai/tuning/audio-qlora.md.txt)
*   [Preference optimization (DPO)](https://gemma.ai/tuning/preference-alignment.md.txt)
*   [Verifiable rewards (GRPO)](https://gemma.ai/tuning/reinforcement-learning.md.txt)
*   [Multi-GPU scaling (FSDP2) & TorchAO QAT](https://gemma.ai/tuning/scaling.md.txt)
*   [Export to vLLM, GGUF, and LiteRT-LM](https://gemma.ai/tuning/export.md.txt)
*   [Unsloth single-GPU swap-in](https://gemma.ai/tuning/unsloth.md.txt)
