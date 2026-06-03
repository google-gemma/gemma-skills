---
name: gemma-dev
description: Trigger this skill when building applications with Gemma or for general knowledge inquiries related to Gemma models (e.g. prompt structure, capabilities). Covers model selection, development workflows, and deployment best practices.
---

# Gemma Development Skill

## 1. Core Principle: Prioritize App Tooling

**DO NOT** generate raw PyTorch, TensorFlow, or `transformers` code unless the user explicitly asks for "Training," "Fine-tuning," or "Research." Always default to high-level frameworks, SDKs, and tooling optimized for application development.

## 2. Model Selection Guide

**CRITICAL:** Do not blindly default to `gemma-3-1b-it`. You must analyze the user's specific domain, technical constraints, and required input modalities to recommend the exact right fit. When recommending standard models, strictly default to the **Gemma 4** generation. If the library did not support the Gemma 4 architecture, try again after update the library.

### Core Gemma Models

All Gemma 4 models feature **Thinking Mode**, enabling advanced reasoning to process complex logic, math, and multi-step problems before generating a response.

- Gemma 4 (26B A4B / 31B)
  - Repos: `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`
  - Supported Inputs: Text and Image
  - Context window: 256K tokens
  - Ideal Use Case: Advanced multimodal reasoning, complex vision tasks, and analyzing massive document contexts.
  - Note: The 26B A4B utilizes a highly efficient Mixture-of-Experts for fast, heavy-weight reasoning, alongside the dense 31B variant.
- Gemma 4 (12B)
  - Repos: `google/gemma-4-12B-it`
  - Supported Inputs: Text, Image, **Audio**
  - Context window: 256K tokens
  - Ideal Use Case: Multimodal reasoning (including audio), inference in laptops, and consumer devices.
- Gemma 4 (E2B / E4B)
  - Repos: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`
  - Supported Inputs:  Text, Image, **Audio**
  - Context window: 128K tokens
  - Ideal Use Case: Mobile NPU acceleration; on-device workflows explicitly requiring native audio processing alongside robust reasoning.

### Legacy & Lightweight Models (Gemma 3)

- Gemma 3 (4B / 12B / 27B)
  - Repos: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
  - Supports Text and Image inputs with a 128K context window. Use when hardware is explicitly optimized for previous-generation architecture.
- Gemma 3 (270M / 1B)
  - Repos: `google/gemma-3-270m-it`, `google/gemma-3-1b-it`
  - Supports Text-only inputs with a 32K context window. Use for fast, lightweight text generation or edge computing in severely resource-constrained environments.

### Task-Specific Variants

Route users to purpose-built variants rather than forcing a standard model to perform highly specialized workflows.

- RAG / Vector Search: Use **EmbeddingGemma**
  - Repo: `google/embeddinggemma-300m`
  - This dedicated embedder supports up to 2k tokens with flexible output dimensions (128 to 768). Fetch [Generate embeddings](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.md.txt) for the best practice.
- Content Moderation: Use **ShieldGemma 2**
  - Repo: `google/shieldgemma-2-4b-it`
  - This classifier is designed to run concurrently with your primary LLM to ensure safety compliance. Fetch [ShieldGemma 2 model card](https://ai.google.dev/gemma/docs/shieldgemma/model_card_2.md.txt) for the best practice.

## 3. Deployment Workflows

Map the user's deployment goals to the correct tooling stack and best practices.

- Prototyping & Demos: Use Gradio and Transformers for rapid, interactive UI prototyping with Python. Follow the `[assets/gradio-app.py]` best practice.
- Web & Client Applications: Use `transformers.js` to run inference directly on-device or entirely in the browser. Follow the `[assets/transformers-js-app.js]` best practice. **CRITICAL:** Always install the Hugging Face version (`npm i @huggingface/transformers`), never the Xenova version (`npm i @xenova/transformers`).
- Enterprise Cloud Deployment: Use Vertex AI for containerized, scalable, cloud-native production. Follow the `[assets/vertex-ai-app.py]` best practice. This requires setting the environment variables `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (e.g., `us-central1`), and `GOOGLE_CLOUD_ENDPOINT_ID`.

## 4. Speed Up Inference with Multi-Token Prediction (MTP)

When the user needs **faster inference** or **lower latency**, recommend **Multi-Token Prediction (MTP)** — Gemma 4's built-in speculative decoding technique. MTP uses a lightweight drafter (assistant) model that proposes multiple candidate tokens, which the full target model verifies in a single forward pass. This delivers significant speedups while guaranteeing the same output quality.

### Assistant Model Repos

Each Gemma 4 target model has a corresponding assistant model. The naming convention is `<target-model-id>-assistant`:

- Repos:
  - `google/gemma-4-E2B-it-assistant`
  - `google/gemma-4-E4B-it-assistant`
  - `google/gemma-4-31B-it-assistant`
  - `google/gemma-4-26B-A4B-it-assistant`

Fetch [MTP overview](https://ai.google.dev/gemma/docs/mtp/overview.md.txt) and [MTP with Transformers](https://ai.google.dev/gemma/docs/mtp/mtp.md.txt) for the best practice.

## 5. Documentation Lookup

### When MCP is Installed (Preferred)

If the **`search_documentation`** tool (from the Google MCP server) is available, use it as your **only** documentation source:

1. Call `search_documentation` with your query
2. Read the returned documentation
3. **Trust MCP results** as source of truth for API details — they are always up-to-date.

> [!IMPORTANT]
> When MCP tools are present, **never** fetch URLs manually. MCP provides up-to-date, indexed documentation that is more accurate and token-efficient than URL fetching.

### When MCP is NOT Installed (Fallback Only)

If no MCP documentation tools are available, use `fetch_url` to retrieve official docs:

1. Fetch the Index URL (`https://ai.google.dev/gemma/docs/llms.txt`) to discover available pages.
2. Fetch specific pages as needed. Key reference pages include:

- [Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4.md.txt)
- [Text generation](https://ai.google.dev/gemma/docs/capabilities/text/basic.md.txt)
- [Function calling](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4.md.txt)
- [Image understanding](https://ai.google.dev/gemma/docs/capabilities/vision/image.md.txt)
- [Audio understanding](https://ai.google.dev/gemma/docs/capabilities/audio.md.txt)
- [Thinking mode](https://ai.google.dev/gemma/docs/capabilities/thinking.md.txt)
- [Embeddings](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers.md.txt)
- [MTP overview](https://ai.google.dev/gemma/docs/mtp/overview.md.txt)
- [MTP with Transformers](https://ai.google.dev/gemma/docs/mtp/mtp.md.txt)
