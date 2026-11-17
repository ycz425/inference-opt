# inference-opt

A benchmarking and configuration-search harness for LLM inference. It measures latency, throughput, memory and quality loss for a model under different inference settings (precision, batch size, compilation), and grid-searches those settings under constraints.

The experiments below use Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct on a single NVIDIA H100 with Hugging Face Transformers.

## Key findings

1. **Eager-mode decoding in HF Transformers is limited by CPU-side kernel launches, not by the GPU.** Time per output token (TPOT) is ~15 ms for *both* Qwen2.5-1.5B (14.9 ms) and Qwen2.5-7B (15.5 ms), even though 7B has ~4.9× the weights. It stays flat from batch 1 to batch 128. At batch 1 that is only ~207 GB/s (1.5B) and ~981 GB/s (7B) of effective weight bandwidth, against the H100's ~3.35 TB/s.
2. **A static KV cache plus CUDA-graph compilation cuts 7B decode latency 2.1×** at batch 1 (15.5 → 7.4 ms/token, ~2.06 TB/s, about 62% of peak HBM bandwidth). The gain shrinks as batch size grows (1.9× at batch 4, 1.6× at batch 16). It is gone at batch 64, and compiled decoding is slower than eager at batch 128 (open question, see below).
3. **bitsandbytes quantization saves memory but is slower than fp16** for a 1.5B model on H100. int8 cuts weight memory 41% at 0.61% CE degradation but is 5.2× slower per token. int4 (NF4) cuts memory 60% but is 1.5× slower and loses 5.1% CE.
4. **Batching is close to free while decoding is overhead-bound.** 1.5B fp16 throughput scales ~61× from batch 1 to batch 64 (67 → 4,071 output tok/s) with no change in TPOT. The cost appears in time to first token (TTFT: 17 → 118 ms).

## How it works

```
config.yaml ──► workload (ShareGPT, fixed-length) ──► preprocessing (chat template, tokenize)
                                                          │
                     backend (HF Transformers: precision / compile) ◄─┘
                                                          │
                benchmark runner ──► TTFT, TPOT, throughput, memory ──► result.json
                quality evaluator ──► CE degradation vs. a reference model
                grid search ──► sweeps inference settings, applies constraints
```

| Package | Responsibility |
|---|---|
| `workload/` | Loads ShareGPT conversations. Keeps only prompts with at least `num_tokens` tokens after the chat template, so every request can be truncated to exactly the same length |
| `preprocessing/` | Applies the chat template and tokenizes. Truncates to `num_tokens` (no padding in fixed-length mode) |
| `backends/` | Backend registry. `TransformersBackend` supports fp32/fp16/bf16, bitsandbytes int8/int4, and `compile` |
| `benchmark/` | Timed benchmark loop, metric aggregation, and cross-entropy quality evaluation |
| `memory/` | Peak and current memory monitors for CUDA and MPS |
| `experiment/` | Runs one config end to end and writes `config.yaml` and `result.json` to a timestamped folder |
| `optimizer/` | Grid search over `precision`, `batch_size` and `compile`, with constraints on memory, latency, throughput and CE degradation |

## Methodology

**Fixed-shape workloads.** Each prompt is a real ShareGPT conversation passed through the model's chat template and truncated to exactly `num_tokens` tokens. Prompts that are too short are skipped rather than padded. Generation uses greedy decoding with `min_new_tokens = max_new_tokens = output_tokens`, so every request produces exactly that many tokens. Every batch therefore does identical work, with no padding, and configs can be compared fairly. For a dense model, speed depends on sequence shape, not token content.

**Warmup.** Before measuring, the first batch is run `warmup_batches` times through the exact calls being measured. This compiles and caches kernels for every shape without using up measured requests. Memory measurement starts after warmup.

**Timing.** For each batch the runner calls `generate(..., 1)` to measure TTFT, then `generate(..., N)` on the same inputs, synchronizing the device around each call.

| Metric | Definition |
|---|---|
| TTFT | Time to generate 1 token (prefill plus the first token). In static batching every request in a batch shares it |
| TPOT | (full-generation time − TTFT) / (N − 1): time per decode step, which is each request's per-token latency |
| E2E latency | Full-generation time of a batch |
| Output tok/s | Σ output tokens / Σ generation time (end to end, includes prefill) |
| Decode tok/s | Σ (output tokens − batch size) / Σ (generation time − TTFT) |
| Prefill tok/s | Σ input tokens / Σ TTFT |
| Model memory | Allocated memory after loading weights |
| Runtime memory | Peak memory − model memory (KV cache + activations) |
| Effective bandwidth | Model memory / median TPOT. Bytes read per decode step ÷ step time. Meaningful at small batch, where weight reads dominate |
| CE degradation | (CE_candidate − CE_reference) / CE_reference × 100, over fixed 512-token ShareGPT sequences, with no padding |

Latencies are reported as medians over batches. Throughputs are ratios of totals rather than means of per-batch rates.

## Results

All runs: single H100, 256 input tokens, 128 output tokens per request.

### Experiment 1: precision × batch size (Qwen2.5-1.5B-Instruct, eager)

64 requests, 1 warmup batch. Quality measured against an fp16 reference on 64 × 512-token sequences (~32k tokens).

| Precision | Batch | Output tok/s | TTFT (ms) | TPOT (ms) | Weights (GB) | Peak (GB) | CE degradation |
|---|---|---|---|---|---|---|---|
| fp16 | 1 | 67 | 17.1 | 14.92 | 3.09 | 3.15 | 0 (reference) |
| fp16 | 4 | 267 | 17.6 | 14.98 | 3.12 | 3.22 | 0 |
| fp16 | 16 | 1,069 | 32.5 | 14.83 | 3.12 | 3.51 | 0 |
| fp16 | 64 | 4,071 | 117.7 | 14.92 | 3.12 | 4.68 | 0 |
| bf16 | 1 | 67 | 17.3 | 14.92 | 3.12 | 3.15 | 0.12% |
| bf16 | 64 | 4,063 | 116.5 | 14.96 | 3.12 | 4.68 | 0.12% |
| int8 | 1 | 13 | 92.7 | 77.53 | 1.81 | 1.91 | 0.61% |
| int8 | 64 | 694 | 358.6 | 90.09 | 1.81 | 4.00 | 0.61% |
| int4 | 1 | 45 | 33.5 | 21.87 | 1.24 | 1.32 | 5.09% |
| int4 | 64 | 1,700 | 929.5 | 30.64 | 1.24 | 3.19 | 5.09% |

The bf16 rows at batch 4 and 16 track fp16 within 1%. Full per-config results are in `experiments/qwen2.5-1.5b_grid_search/`.

**Observations**
- **TPOT is flat across batch size and precision (fp16/bf16).** That is the signature of a fixed per-step cost: Python and kernel-launch overhead. The weights would stream in ~0.9 ms at full bandwidth.
- **bf16 vs fp16 is a noise floor, not a loss.** The same speed and 0.12% CE degradation give a reference point for "harmless" precision changes.
- **LLM.int8 is nearly lossless (0.61%, ~5× the noise floor) but slow.** Its outlier decomposition splits every matmul into int8 and fp16 parts.
- **NF4 is too lossy for a 1.5B model (5.1%)** and its prefill is ~8× slower than fp16 at batch 64 (17.6k vs 139k tok/s), from dequantizing weights on every matmul.
- **Quantization doesn't speed up an overhead-bound decode.** Reading fewer bytes doesn't help when bytes aren't the bottleneck, and dequantization adds work.

**Grid-search outcome** (constraints: CE ≤ 1%, peak memory ≤ 8 GiB, mean E2E ≤ 10 s; objective: output tok/s): **fp16, batch 64, 4,071 tok/s.** bf16 ties within noise. int4 fails the quality limit, and int8 at batch ≥ 4 fails the latency limit.

### Experiment 2: eager vs compiled (Qwen2.5-7B-Instruct, bf16)

128 requests, 2 warmup batches. "Compiled" means a static KV cache plus Transformers' built-in compilation of the decode step (`CompileConfig(fullgraph=True, mode="reduce-overhead")`, which uses CUDA graphs). Prefill stays eager. Generated text is coherent and nearly identical between modes.

| Batch | Eager TPOT (ms) | Compiled TPOT (ms) | Speedup | Eager output tok/s | Compiled output tok/s | Eager eff. BW | Compiled eff. BW |
|---|---|---|---|---|---|---|---|
| 1 | 15.52 | **7.39** | **2.10×** | 64 | 135 | 981 GB/s | 2,064 GB/s |
| 4 | 15.41 | 7.93 | 1.94× | 257 | 490 | 990 GB/s | 1,924 GB/s |
| 16 | 15.39 | 9.77 | 1.58× | 994 | 1,511 | — | — |
| 64 | 15.58 | 15.24 | 1.02× | 3,434 | 3,499 | — | — |
| 128 | 16.03 | 23.49 | 0.68× | **5,755** | 4,312 | — | — |

Effective bandwidth is shown only at small batch, where weight reads dominate each step. TTFT is unchanged by compilation (17–20 ms at batch 1, ~406 ms at batch 64, ~812–816 ms at batch 128), because only decode is compiled. Prefill runs at ~40k tok/s. Full results are in `experiments/qwen2.5-7b_compile/`.

**Observations**
- **Model size barely changes eager TPOT** (14.9 ms at 1.5B, 15.5 ms at 7B). Both models have 28 layers and launch about the same number of kernels per step. In eager mode, step time is the larger of CPU launch time and GPU work, and launch time wins.
- **CUDA graphs remove the launch cost.** Compiled 7B at batch 1 reaches ~62% of peak memory bandwidth, so decode becomes memory-bound, as expected for a 15 GB model at batch 1.
- **With overhead removed, real GPU work shows up.** Compiled TPOT rises with batch size (7.4 → 7.9 → 9.8 ms), while eager TPOT hides it behind the fixed launch cost.
- **Compilation is a latency optimization here, not a throughput one.** The highest throughput is still eager at batch 128.

## Open questions and caveats

- **Why compiled decoding is slower at batch 128.** Candidates: (a) CUDA graphs skipped or recompiled at large shapes (compiled batch-64 TPOT almost exactly matches eager, which is suspicious); (b) the static cache is preallocated to the maximum length and masked, so attention reads the full cache every step while the dynamic cache reads only the tokens so far, which costs more as batch grows; (c) Inductor-generated kernels underperforming cuBLAS/SDPA at large shapes. Compiled runtime memory is also higher (10.3 vs 6.6 GB at batch 128). Next step: check the run logs for `skipping cudagraphs` or recompiles.
- **Few samples at large batch.** Batch 64 and 128 cover only 1–2 measured batches (128 requests ÷ batch size). Variation between batches is under ~2% elsewhere, but treat those cells as single measurements.
- **One compiled batch-16 outlier** (TTFT 308 ms vs ~108 ms) suggests a recompile during measurement. The medians are unaffected.
- **Single runs.** Each config was run once. On the H100, variation within a run was ~1%, so the large effects above are well beyond noise. Small differences (such as bf16 vs fp16 speed) are not.
- **Truncated prompts.** Truncation to a fixed length removes the end of the chat template, so prompts end mid-message. This doesn't affect speed, but the generated text is not representative of real replies.
- **Quality is measured on ShareGPT conversation text** (mostly user turns), not on assistant responses or downstream tasks.

## Usage

### Install

```bash
pip install -r requirements.txt
```

The code uses Python 3.12 `type` aliases (`config.py`, `workload/base.py`). On Python 3.11, replace `type X = ...` with `X = ...`. Triton (needed for `compile` on CUDA) is Linux-only. bitsandbytes int8/int4 requires CUDA.

### Run one config

```bash
python main.py -c config.yaml -e experiments
```

### Run a grid search

Edit the `grid`, `constraint` and output folder at the bottom of `optimizer/grid_search.py`, then:

```bash
python -m optimizer.grid_search
```

### Example config

```yaml
inference_config:
  model_name: Qwen/Qwen2.5-7B-Instruct
  backend: transformers
  precision: bf16          # fp32 | fp16 | bf16 | int8 | int4
  device: cuda             # cuda | mps
  batch_size: 16
  compile: true            # static KV cache + CUDA-graph decode

workload_config:
  dataset: sharegpt
  num_requests: 128        # must be a multiple of batch_size
  num_tokens: 256          # fixed prompt length (null = natural lengths, padded)
  seed: 42

output_tokens: 128
warmup_batches: 2

quality_config:            # or null to skip
  reference:
    model_name: Qwen/Qwen2.5-7B-Instruct
    backend: transformers
    precision: bf16
    device: cuda
    batch_size: 16
  workload:
    dataset: sharegpt
    num_requests: 64
    num_tokens: 512
    seed: 123
```

Each run writes `<experiment dir>/<timestamp>/config.yaml` and `result.json`, with per-batch timings and output texts.
