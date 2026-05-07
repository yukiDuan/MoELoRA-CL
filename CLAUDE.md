# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoELoRA continual learning experiment: implements Mixture-of-Experts LoRA (parameter-efficient fine-tuning with multiple LoRA experts and a learned router) applied to sequential task learning on a causal LM (default: Llama-2-7b).

Based on the paper: "MoELoRA: Contrastive Learning Guided Mixture of Experts on Parameter-Efficient Fine-Tuning for Large Language Models" (Luo et al., 2024).

## Running

```bash
cd MoELoRA
pip install -r requirements.txt
python run.py
```

Requires `torch>=2.0.0` and `transformers>=4.36.0`. Needs access to `meta-llama/Llama-2-7b-hf` (or change `model_name` in `DEFAULT_CONFIG` in `utils.py`).

## Architecture

All code lives in `MoELoRA/`:

- **`run.py`** — Entry point. Orchestrates the full pipeline: load model, wrap with MoELoRA, train sequentially on 3 tasks, report forgetting metrics, run inference.
- **`modeling.py`** — Model definitions:
  - `LoRAExpert` — Single low-rank adapter (A·B with scaling).
  - `MoELoRALayer` — Replaces an `nn.Linear` with frozen base + N experts + top-k router. Computes load-balance loss.
  - `MoELoRAModel` — Wraps a HuggingFace `PreTrainedModel`, freezes base params, replaces target linear layers (default: `q_proj`, `v_proj`) with `MoELoRALayer`.
  - `contrastive_loss_placeholder` — InfoNCE stub (returns 0); interface reserved for routing-distribution contrastive loss.
- **`trainer.py`** — Training loop and evaluation:
  - `MoELoRATrainer` — Handles single-task and continual (sequential) training with combined loss: task_loss + λ_lb·load_balance + λ_cl·contrastive.
  - `ForgettingMonitor` — Tracks per-task eval metrics across training stages; computes BWT (backward transfer) forgetting metric.
- **`utils.py`** — Config, data, and helpers:
  - `DEFAULT_CONFIG` — All hyperparameters in one dict.
  - `load_model_and_tokenizer` — Loads base model in bfloat16.
  - `TASK_DATA` / `create_continual_learning_tasks` — Three synthetic tasks (translation, QA, summarization) for demo purposes.

## Key Design Decisions

- Base model parameters are fully frozen; only expert weights and router weights are trainable.
- Router uses softmax over all experts, then selects top-k (default 2) with renormalized gating.
- Load-balance loss follows Switch Transformer formulation: `N * (frequency · probability)`.
- Contrastive loss (`lambda_cl`) is currently disabled (0.0) with a placeholder function — the interface is ready for implementation.
- Labels use `-100` for padding tokens (standard HuggingFace ignore index).

## Language

Code comments and print output are in Chinese (简体中文). Maintain this convention.
