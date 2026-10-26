# Current Learning Progress

Last updated: 2026-10-26

## Project Goal

Build and understand a Qwen3 inference stack from tensor operations through serving-oriented infrastructure. The target scope is the complete tiny-llm guide through Week 3, including KV cache, quantized inference, Flash Attention, batching, paged attention, and MoE. Correctness comes first; performance claims must be backed by reproducible benchmarks.

## Primary Development Repository

- Independent project: `KazeBox33/qwen3-inference-from-scratch`, branch `main`
- Learning fork: `KazeBox33/tiny-llm`
- Current source branch: `complete-week1-boundaries` (through `01031c7`)
- Fork `main` is the course template, not the personal implementation branch.
- The entries below marked historical describe the earlier course, not revised Week 2 completion.

The repositories have separate histories. New learning commits are cherry-picked with source attribution into the independent project; no force push or course-template overwrite is used.

## Revised Course Checkpoint

| Area | Status | Evidence / remaining boundary |
| --- | --- | --- |
| Week 1 boundary handling | Implemented; focused regressions pass | Causal shape validation, top-p threshold, generation limits and text-buffer flushing |
| Day 1 KV storage / append | Tested | Initial values, history preservation, shapes, offsets and mask pass-through |
| Request-scoped model caches | Connected | Independent caches per layer/request; invalid offsets rejected before layer mutation |
| Cached generation | Connected | One full-prompt prefill, then single-token calls; EOS/limits/input checks and final flush |
| Tiny-model correctness | Locally verified | Full sequence, single-token and 3/1/2 chunks compared with MLX-LM |
| SiLU reference consistency | Fixed; local operator regression passes | Compiled `x * mx.sigmoid(x)` matches the reference path; no new custom fused kernel |
| Real 4B prefill regression | Source notes report passing; skipped on this Mac | Local weights unavailable; no download performed |
| Revised dense/cache-only baseline | Pending alignment | Current model retains earlier quantized projections; not an isolated cache-only benchmark |
| Capacity-bounded cache / revised entrypoints | Not implemented by these commits | Do not infer completion from request isolation or a generation token limit |
| Revised benchmark/profile and later kernels | Pending | Keep legacy experiments separate; no new throughput or memory claims |

Synced source commits: `d9c7f5b`, `e5f8054`, `4478291`, `01031c7`.
Detailed source-side experiments and numerical diagnosis: [cache flow notes](../../notes/week2-day1-cache-flow.md).

Local verification on 2026-10-26 in the installed project environment:

```bash
HF_HUB_OFFLINE=1 python -m pytest -q -rs \
  tests/test_week_1_completion.py tests/test_week_2_cache.py \
  tests/test_week_2_generation.py tests/test_week_2_model.py \
  tests/test_week_2_numerics.py
```

Result: **39 passed, 3 skipped**. The three skipped cases are the seed 0/1/7 Qwen3-4B prefill tests, not successful model checks.
No extension source changed in this sync, but this checkout initially lacked built binaries: the first run had 36 passed, 3 failed and 3 skipped because `quantized_matmul` was unavailable.
Rebuilt the existing extension locally without changing source or dependencies. CMake initially selected Xcode Python 3.9; clearing its cached Python discovery and explicitly selecting this project's Python 3.12 fixed the build.
From `src/extensions`, the successful command was:

```bash
DEBUG=0 CMAKE_ARGS="-U*Python* -DPython_EXECUTABLE=$VIRTUAL_ENV/bin/python" python build.py
```

Activate the project's virtual environment first so `VIRTUAL_ENV` and `python` refer to it. Build outputs remain ignored and are not committed.

## Immediate Next Step

Continue from the revised Week 2 Day 1 guide: review cache state flow and align the dense readable baseline before benchmarking.
The current fork and this mirror still retain older guide/CLI/dependency interfaces; do not copy revised `--week2-checkpoint` commands without first implementing and verifying that interface.
Then follow revised benchmarking/profiling order, choosing optimizations from measurements rather than jumping to the historical Metal Flash Attention task.
Retain the source notes' capacity-cache follow-up as pending; a generation `max_tokens` bound is not allocated KV-cache capacity management.

## Historical Repository Layout (2026-07)

- Repository: `KazeBox33/tiny-llm`
- Branch: `main`
- Upstream course: `skyzh/tiny-llm`
- Resume-facing mirror: `KazeBox33/qwen3-inference-from-scratch`

This was the July setup; use the current source branch and revised checkpoint above when resuming.

## Historical Guide Progress (2026-07)

| Guide section | Status | Evidence |
| --- | --- | --- |
| Week 1: model fundamentals | Complete | Attention, RoPE, GQA, RMSNorm, MLP, model loading, generation, and sampling implemented and tested |
| Week 2 Day 1: KV cache | Complete | KV-cache decoding and Week 2 model path implemented and tested |
| Week 2 Day 2-3: quantized matmul | Complete | CPU primitive, Metal kernel, model integration, and benchmark reports committed |
| Week 2 Day 4-5 Task 1: Flash Attention wrapper | Complete | Commit `59ddd81`; Python shape/dtype adapter implemented |
| Week 2 Day 4-5 Task 2: CPU Flash Attention | Complete | Commit `83a768e`; 9 targeted tests pass |
| Week 2 Day 4-5 Task 3: GPU Flash Attention | Next | Metal kernel, GPU dispatch, and metallib registration not implemented yet |
| Week 2 Day 4-5 Task 4: model integration | Pending | `use_flash_attention` exists, but attention execution still always uses the grouped fallback |
| Week 2 Day 6-7: prefill and batching | Pending | Start only after Flash Attention Task 4 passes |
| Week 3: paged attention and MoE | Pending | Start after Week 2 is complete |

## Historical Flash Attention Checkpoint

The CPU implementation in `src/extensions/src/flash_attention.cpp` is complete and understood. Topics already studied:

- tiled attention with `Br = Bc = 32`;
- `Tr` and `Tc` tile counts;
- flattened batch/head dimension `N`;
- GQA query-head to KV-head mapping;
- causal alignment when `L != S`;
- fully masked tile skipping and fully valid tile optimization;
- online-softmax state `m_i`, `l_i`, and `o_i`;
- rescaling prior softmax/output state when a later tile introduces a larger maximum;
- MLX lazy primitives, command encoders, lambda captures, and `unsafe_weak_copy`;
- contiguous memory, raw pointers, and manual flattened indexing.

Verified on 2026-07-19:

```text
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 4 -- -k task_2

9 passed, 11 deselected
```

## Historical Next Task (Not the Current Resume Point)

Follow `book/src/week2-04-flash-attention.md`, Task 3, in this order:

1. Learn the Metal kernel argument list and memory address spaces: `device`, `constant`, `threadgroup`, and register-local values.
2. Map CPU loops to GPU execution: `(n, i)` to the threadgroup grid, query row `a` to `simd_gid`, and KV column `b` to `simd_lid`.
3. Implement `src/extensions/src/flash_attention.metal` with one threadgroup per `(head, query tile)`.
4. Keep Q and partial O in threadgroup memory; keep online-softmax scalars in registers.
5. Use `simd_max` and `simd_sum` for row reductions while iterating KV tiles inside the threadgroup.
6. Implement `FlashAttention::eval_gpu(...)` in `src/extensions/src/flash_attention.cpp`.
7. Add `src/flash_attention.metal` to `mlx_build_metallib(...)` in `src/extensions/CMakeLists.txt`.
8. Rebuild and run Task 3 tests:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 4 -- -k task_3
```

Do not start Task 4 model integration until Task 3 correctness and causal-mask performance tests pass.

## Historical New Mac Setup

This project requires an Apple Silicon Mac for MLX/Metal execution.

Clone the primary development repository:

```bash
git clone https://github.com/KazeBox33/tiny-llm.git
cd tiny-llm
```

Install PDM if it is not already available:

```bash
brew install pdm
```

Install Xcode from the App Store, select its developer directory if needed, and install the Metal toolchain:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
```

Install the Python environment and verify MLX:

```bash
pdm install -v
pdm run check-installation
```

Rebuild the custom C++/Metal extension and verify the current checkpoint:

```bash
DEBUG=0 pdm run build-ext
DEBUG=0 pdm run test --week 2 --day 4 -- -k task_2
```

The Task 3 extension tests do not require model weights. Download `Qwen/Qwen3-0.6B-MLX-4bit` only when moving on to Task 4 end-to-end generation and benchmarking.

For VS Code, install the Python/Pylance extensions and clangd. Open the repository root so `.clangd` and generated compile commands can be discovered.

## Working Rules

- Read the matching guide chapter before each task.
- Teach the systems concept, data flow, shapes, and syntax before implementing AI-infrastructure code.
- Compare against `src/tiny_llm_ref` and `src/extensions_ref` for correctness checks.
- Complete one guide task, run its smallest targeted tests, then commit and push.
- Keep performance measurements under `benchmarks/` and interpretations under `project-notes/benchmarks/`.
- Keep resume-ready evidence in `project-notes/resume/RESUME_NOTES.md`.
