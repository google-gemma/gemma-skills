---
name: gemma-app-builder
description: Trigger this skill for end-to-end application development with the Gemma model family. Covers precise model selection for multimodal, RAG, and agent workflows. Provides deployment architectures utilizing Gradio, Ollama, LM Studio, AnythingLLM, OpenWebUI, LiteRT-LM, transformers.js, and Vertex AI.
---

# Gemma App Builder

This documentation outlines the structured approach for building, integrating, and deploying applications using the Gemma family of models. 

## Core Principle: Prioritize App Tooling

**DO NOT** generate raw PyTorch, TensorFlow, or `transformers` code unless the user explicitly asks for "Training," "Fine-tuning," or "Research." Always default to high-level frameworks, SDKs, and tooling optimized for application development.

## Model Selection Guide

**CRITICAL:** Do not blindly default to `gemma-3-1b-it`. You must analyze the user's specific domain, technical constraints, and required input modalities to recommend the exact right fit. When recommending standard models, strictly default to the **Gemma 4** generation.

### 1. Core Gemma Models 

Always recommend the Gemma 4 family for new applications unless specific hardware constraints dictate otherwise. Use the matrices below to match the model to the user's requirements.

**Gemma 4 (Latest Generation)**

All Gemma 4 models feature **Thinking Mode**, enabling advanced reasoning to process complex logic, math, and multi-step problems before generating a response.

| Model Variant | Supported Inputs | Context | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| **Gemma 4 (26B-A4B / 31B)** | Text & Image | 256k | Advanced multimodal reasoning, complex vision tasks, and analyzing massive document contexts. The 26B-A4B utilizes a highly efficient **Mixture-of-Experts** for fast, heavy-weight reasoning, alongside the dense 31B variant. |
| **Gemma 4 (E2B / E4B)** | Text, Image, **Audio** | 128k | Mobile NPU acceleration; on-device workflows explicitly requiring native **audio** processing alongside robust reasoning capabilities. |

**Gemma 3 (Legacy & Lightweight)**

| Model Variant | Supported Inputs | Context | Ideal Use Case |
| :--- | :--- | :--- | :--- |
| **Gemma 3 (4B / 12B / 27B)** | Text & Image | 128k | Standard multimodal reasoning; use when hardware is optimized for previous-generation architecture. |
| **Gemma 3 (270M / 1B)** | Text only | 32k | Fast, lightweight text generation; edge computing in severely resource-constrained environments. |

### 2. Task-Specific Variants

Do not force a standard model to perform highly specialized workflows. Route the user to these purpose-built variants based on their goal.

| User Goal | ❌ Avoid | ✅ Use Variant | Rationale |
| :--- | :--- | :--- | :--- |
| **Long-Form Summarization** (e.g., "Summarize this PDF") | Standard Gemma (Decoder-only) | **T5Gemma 2 (270M/1B/4B)** | Encoder-decoder architecture is vastly superior for compressing long-context inputs. |
| **Healthcare & Imaging** (e.g., DICOM, CT/MRI) | Standard Gemma | **MedGemma 1.5** | Pre-trained specifically for 3D medical imaging and Electronic Health Record (EHR) data. |
| **Agentic Workflows** (e.g., Tool use, JSON output) | Standard Prompting | **FunctionGemma** | Fine-tuned exclusively for reliable tool execution and structured data generation. |
| **RAG / Vector Search** | Gecko / BERT | **EmbeddingGemma** | Dedicated embedder supporting up to 2k tokens with flexible output dimensions (128 to 768). |
| **Content Moderation** | Python Rule Scripts | **ShieldGemma 2** | Classifier model designed to run concurrently with your primary LLM to ensure safety compliance. |

## Deployment Workflows

Map the user's deployment goals to the correct tooling stack and documentation references. 

*   **Prototyping & Demos**
    *   **Goal:** Rapid, interactive UI prototyping with Python.
    *   **Tooling:** Gradio, Transformers
    *   **Docs:** `[references/gradio.md]`
*   **Web & Client Applications**
    *   **Goal:** Running inference directly on-device or entirely in the browser.
    *   **Tooling:** transformers.js
    *   **Docs:** `[references/transformers-js.md]`
*   **Enterprise Cloud Deployment**
    *   **Goal:** Containerized, scalable, cloud-native production.
    *   **Tooling:** Vertex AI
    *   **Docs:** `[references/vertex-ai.md]`
