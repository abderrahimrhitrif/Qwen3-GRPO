# Qwen3-4B GRPO Fine-Tuning (Kernel)

> **Note:** The `Qwen3-GRPO` repository is empty. The actual code and notebook reside in the `kernel` repository (`nb/Qwen3_(4B)-GRPO.ipynb`).

## Overview
This project demonstrates parameter-efficient fine-tuning (PEFT/LoRA) of a Qwen3 model using Group Relative Policy Optimization (GRPO) via Unsloth and Hugging Face TRL, transforming a base language model into a reasoning model.

## Architecture & Pipeline
1. **Model Loading**: Loads Qwen3 base model with Unsloth fast inference optimizations.
2. **Pre-FineTuning**: Primes the model on a subset of the Open Math Reasoning dataset using SFTTrainer to learn custom reasoning tags (`<start_working_out>`, `<end_working_out>`, `<SOLUTION>`).
3. **GRPO Training**: Trains the model using GRPOTrainer with custom reward functions checking format adherence, numerical correctness, and algebraic ratio tolerances.
4. **Export**: Saves adapters and exports models to GGUF / Ollama formats.

## Data
- **Dataset**: OpenR1 Math dataset (`open-r1/DAPO-Math-17k-Processed`) and Open Math Reasoning dataset.

## Method & Algorithms
- **GRPO (Group Relative Policy Optimization)**: Reinforcement learning fine-tuning optimizing completion groups without a separate critic model.
- **LoRA (Low-Rank Adaptation)**: PEFT parameter updates targeting attention and MLP projection weights (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).

## Verified Results & Metrics
- Successfully primed model formatting and executed GRPO training steps with reward convergence tracking.

## Limitations & Caveats
- Training demonstrated on consumer / T4 GPU instances with small subset epochs; reasoning capability depends on training duration.

## Run Instructions
Open and run the Jupyter notebook located in the `kernel` repository:
```bash
# Notebook path in kernel repo:
nb/Qwen3_(4B)-GRPO.ipynb
```
