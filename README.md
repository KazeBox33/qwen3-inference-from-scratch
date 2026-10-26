# Qwen3 Inference from Scratch

This repository records my implementation of a Qwen3-style LLM inference stack from low-level tensor operations. The goal is to understand how modern decoder-only LLM inference works by building the core components directly instead of only calling high-level model APIs.

The project is based on the excellent [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) course, with my own step-by-step implementation, notes, tests, and future experiments.

## Current Learning Checkpoint (2026-10-26)

Current work follows the revised Week 2 KV-cache chapter while retaining the earlier course implementation and experiments.
The latest changes are synchronized from `KazeBox33/tiny-llm`, branch `complete-week1-boundaries`, through source commit `01031c7`.
This independent repository retains its own commit history; it is not a replacement for the upstream course template.

- Added regression tests for initial KV storage and history-preserving appends.
- Connected per-request, per-layer cache creation, offset validation, and bounded cached generation with final text-buffer flushing.
- Verified full-sequence, token-by-token, and chunked execution on a tiny BF16/4-bit Qwen3 model.
- Matched the MLX reference SiLU rounding path and added optional real-Qwen3-4B prefill regressions.
- The current Week 2 model still uses the existing quantized projections. It is not yet the revised chapter's dense, cache-only performance baseline, and no new speedup is claimed.

Local verification on this Mac: **39 passed, 3 skipped**; the skipped cases require locally cached Qwen3-4B weights.
The source notes record successful 4B checks on another environment; those are not presented as results reproduced here.
See [current progress](project-notes/learning/CURRENT_PROGRESS.md) and [cache flow / numerical diagnosis](notes/week2-day1-cache-flow.md).

Next: finish aligning the revised Day 1 baseline and entrypoints, then benchmark/profile it before choosing the next kernel optimization.

## What This Project Covers

Implemented so far:

- Basic matrix APIs used by the model path
- Scaled dot-product attention
- Multi-head attention
- Rotary positional encoding, including Qwen3 non-traditional RoPE
- Grouped Query Attention
- Causal attention masking for both training-style and KV-cache-style shapes
- Qwen3 attention block with Q/K RMSNorm and RoPE
- RMSNorm with float32 accumulation
- Numerically stable SiLU
- Qwen3 SwiGLU MLP
- Qwen3 model loading, generation, and sampling
- KV-cache based autoregressive decoding
- 4-bit quantized embedding and linear layers
- Custom MLX C++/Metal quantized matrix multiplication
- Flash Attention Python wrapper and tiled CPU online-softmax primitive

Remaining work (the order follows the revised guide, not the legacy chapter numbers):

- Revised Week 2 baseline and benchmark/CLI alignment
- Metal Flash Attention kernel and model integration
- Prefill, chunked prefill, and continuous batching
- Paged attention and KV-cache memory management
- Mixture-of-Experts inference

## Why This Project

I am using this project to learn the internals of LLM inference:

- How Q/K/V projections become attention heads
- Why GQA reduces KV memory and bandwidth
- How RoPE injects token position into Q/K vectors
- Why causal masks differ between full prefill and cached decoding
- Where low precision is safe and where float32 accumulation is useful
- How Qwen3-style blocks combine RMSNorm, attention, residuals, and SwiGLU MLPs

This is intended as a learning-oriented implementation that can grow into a compact inference engine.

## Current Test Status

Latest focused regression (no model downloads; run in an installed project environment):

```bash
HF_HUB_OFFLINE=1 python -m pytest -q -rs \
  tests/test_week_1_completion.py tests/test_week_2_cache.py \
  tests/test_week_2_generation.py tests/test_week_2_model.py \
  tests/test_week_2_numerics.py
```

Result on 2026-10-26: **39 passed, 3 skipped**. This is a focused regression, not a claim that all revised course tests pass.

Historical Week 1 chapter commands:

```bash
pdm run test --week 1 --day 1
pdm run test --week 1 --day 2
pdm run test --week 1 --day 3
pdm run test --week 1 --day 4
```

Historical results from the earlier course version (not rerun in this sync):

```text
Week 1 Day 3: 60 passed
Week 1 Day 4: 22 passed
Week 2 Day 4 Task 2 (CPU Flash Attention): 9 passed
```

## Setup

Install dependencies:

```bash
pdm install
```

Check the environment:

```bash
pdm run check-installation
```

Run tests for a specific chapter:

```bash
pdm run test --week 1 --day 4
```

## Repository Layout

```text
src/tiny_llm/
  attention.py              attention, GQA, causal mask
  positional_encoding.py    RoPE
  layer_norm.py             RMSNorm
  basics.py                 linear, softmax, SiLU
  qwen3_week1.py            Qwen3 attention, MLP, transformer/model path
```

Reference implementations and tests from the course are kept in:

```text
src/tiny_llm_ref/
tests_refsol/
book/
```

Project notes written during implementation are organized in:

```text
project-notes/
  learning/        current progress and cross-machine handoff
  resume/          resume-ready summaries and measurable results
  benchmarks/      benchmark reports and performance interpretation
  optimization/    future optimization backlog and bottleneck analysis
```

## Attribution

This project is built while following [tiny-llm - LLM Serving in a Week](https://skyzh.github.io/tiny-llm/) by skyzh. The original course repository is [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm).

My work in this repository focuses on implementing, understanding, testing, and documenting the inference components step by step.
