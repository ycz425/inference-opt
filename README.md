# inference-opt

A benchmarking and configuration-search harness for LLM inference. It measures latency, throughput, memory and quality loss for a model under different inference settings (precision, batch size, compilation), and grid-searches those settings under constraints.

The experiments below use Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct on a single NVIDIA H100, with Hugging Face Transformers and vLLM as backends.

## Key findings

1. **Eager-mode decoding in HF Transformers is limited by CPU-side kernel launches, not by the GPU.** Time per output token (TPOT) is ~15 ms for *both* Qwen2.5-1.5B (14.9 ms) and Qwen2.5-7B (15.5 ms), even though 7B has ~4.9× the weights. It stays flat from batch 1 to batch 128. At batch 1 that is only ~207 GB/s (1.5B) and ~981 GB/s (7B) of effective weight bandwidth, against the H100's ~3.35 TB/s.
2. **A static KV cache plus CUDA-graph compilation cuts 7B decode latency 2.1×** at batch 1 (15.5 → 7.4 ms/token, ~2.06 TB/s, about 62% of peak HBM bandwidth). The gain shrinks as batch size grows (1.9× at batch 4, 1.6× at batch 16). It is gone at batch 64, and compiled decoding is slower than eager at batch 128 (open question, see below).
3. **bitsandbytes quantization saves memory but is slower than fp16** for a 1.5B model on H100. int8 cuts weight memory 41% at 0.61% CE degradation but is 5.2× slower per token. int4 (NF4) cuts memory 60% but is 1.5× slower and loses 5.1% CE.
4. **Batching is close to free while decoding is overhead-bound.** 1.5B fp16 throughput scales ~61× from batch 1 to batch 64 (67 → 4,071 output tok/s) with no change in TPOT. The cost appears in time to first token (TTFT: 17 → 118 ms).
5. **vLLM has the lowest single-request latency, but an anomalous large-batch slowdown in this setup.** At batch 1, vLLM (compiled + CUDA graphs) decodes 7B at 6.40 ms/token, 1.15× faster than compiled HF and 2.4× faster than eager HF (~2.38 TB/s, ~71% of peak HBM bandwidth). Even with compilation off, vLLM beats eager HF 1.6×. From batch 16 up, vLLM's decode step grows ~0.65 ms per sequence, and at batch 128 it delivers 1,258 tok/s against HF eager's 5,755. vLLM's own `vllm bench latency` reproduces the same numbers to within 0.1%, so the harness is not the cause (open question, see below).

## How it works

```
config.yaml ──► workload (ShareGPT, fixed-length) ──► preprocessing (chat template, tokenize)
                                                          │
              backend (HF Transformers | vLLM: precision / compile) ◄─┘
                                                          │
                benchmark runner ──► TTFT, TPOT, throughput, memory ──► result.json
                quality evaluator ──► CE degradation vs. a reference model
                grid search ──► sweeps inference settings, applies constraints
```

| Package | Responsibility |
|---|---|
| `workload/` | Loads ShareGPT conversations. Keeps only prompts with at least `num_tokens` tokens after the chat template, so every request can be truncated to exactly the same length |
| `preprocessing/` | Applies the chat template and tokenizes. Truncates to `num_tokens` (no padding in fixed-length mode) |
| `backends/` | Backend registry. `TransformersBackend` supports fp32/fp16/bf16, bitsandbytes int8/int4, and `compile`. `vLLMBackend` supports fp32/fp16/bf16 on CUDA, with `compile` mapped to vLLM's torch.compile + CUDA graphs (`enforce_eager=not compile`) |
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

The bf16 rows at batch 4 and 16 track fp16 within 1%. Full per-config results are in `runs/qwen2.5-1.5b_grid_search/`.

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

Effective bandwidth is shown only at small batch, where weight reads dominate each step. TTFT is unchanged by compilation (17–20 ms at batch 1, ~406 ms at batch 64, ~812–816 ms at batch 128), because only decode is compiled. Prefill runs at ~40k tok/s. Full results are in `runs/qwen2.5-7b_compile/`.

**Observations**
- **Model size barely changes eager TPOT** (14.9 ms at 1.5B, 15.5 ms at 7B). Both models have 28 layers and launch about the same number of kernels per step. In eager mode, step time is the larger of CPU launch time and GPU work, and launch time wins.
- **CUDA graphs remove the launch cost.** Compiled 7B at batch 1 reaches ~62% of peak memory bandwidth, so decode becomes memory-bound, as expected for a 15 GB model at batch 1.
- **With overhead removed, real GPU work shows up.** Compiled TPOT rises with batch size (7.4 → 7.9 → 9.8 ms), while eager TPOT hides it behind the fixed launch cost.
- **Compilation is a latency optimization here, not a throughput one.** The highest throughput is still eager at batch 128.

### Experiment 3: vLLM vs HF Transformers (Qwen2.5-7B-Instruct, bf16)

Same workload as Experiment 2 (128 requests, 2 warmup batches), vLLM 0.25.0 (V1 engine, FlashAttention 3). **Static batching:** each batch is one `LLM.generate()` call with `max_num_seqs = batch_size`. Since every request has the same input and output length, vLLM's continuous batching admits the whole batch together and all requests finish on the same step, which makes it equivalent to HF's static batching. Prefix caching is off (otherwise the second `generate` call on the same prompts would skip prefill and break the TTFT/TPOT method), and `max_num_batched_tokens ≥ batch_size × input length`, so prefill runs in one step without chunking. Output length is forced with `ignore_eos` and `min_tokens = max_tokens`. "Compiled" is vLLM's default (torch.compile + CUDA graphs). "Eager" sets `enforce_eager=True`.

TPOT (ms):

| Batch | HF eager | HF compiled | vLLM eager | vLLM compiled |
|---|---|---|---|---|
| 1 | 15.52 | 7.39 | 9.81 | **6.40** |
| 4 | 15.41 | **7.93** | 9.84 | 8.38 |
| 16 | 15.39 | **9.77** | 16.68 | 16.33 |
| 64 | 15.58 | **15.24** | 47.88 | 48.28 |
| 128 | **16.03** | 23.49 | 87.41 | 89.25 |

Output tok/s and TTFT:

| Batch | HF eager tok/s | HF compiled tok/s | vLLM eager tok/s | vLLM compiled tok/s | vLLM TTFT (ms) |
|---|---|---|---|---|---|
| 1 | 64 | 135 | 101 | **154** | 19.5 |
| 4 | 257 | **490** | 390 | 454 | 64.0 |
| 16 | 994 | **1,511** | 868 | 884 | 242 |
| 64 | 3,434 | **3,499** | 1,166 | 1,158 | 943 |
| 128 | **5,755** | 4,312 | 1,258 | 1,236 | 1,923 |

Memory isn't reported for vLLM: the engine runs in a separate process, so the harness's `torch.cuda` memory monitor can't see its allocations. Full results are in `runs/qwen2.5-7b_vllm/`.

**Observations**
- **vLLM is the fastest backend at batch 1** (6.40 ms/token, ~2.38 TB/s effective bandwidth vs ~2.06 TB/s for compiled HF). Compiled HF overtakes it at batch 4.
- **vLLM's eager mode is already much faster than HF's** (9.8 vs 15.5 ms at batch 1). Its own model implementation and fused kernels launch far fewer kernels per step, so even without CUDA graphs it is less launch-bound.
- **vLLM's step time grows linearly with batch size:** about 6 ms + 0.65 ms per sequence, identical with and without compilation from batch 16 up. Real decode work for a 7B model at ~300 tokens of context should add well under 0.1 ms per sequence. Prefill is also slow, at ~17k tok/s vs HF's ~40k (TTFT ≈ 15 ms per request).
- **The slowdown is reproduced outside the harness.** A standalone diagnostic (`diagnostics/vllm_diag.slurm`) ran vLLM's own `vllm bench latency` with the same shape: 2.320 s at batch 16 and 13.218 s at batch 128, matching the harness's 2.321 s and 13.222 s. Removing `min_tokens`, switching greedy vs random sampling, and enabling chunked prefill all made no difference. The GPU reported no throttling.

## Open questions and caveats

- **Why compiled decoding is slower at batch 128.** Candidates: (a) CUDA graphs skipped or recompiled at large shapes (compiled batch-64 TPOT almost exactly matches eager, which is suspicious); (b) the static cache is preallocated to the maximum length and masked, so attention reads the full cache every step while the dynamic cache reads only the tokens so far, which costs more as batch grows; (c) Inductor-generated kernels underperforming cuBLAS/SDPA at large shapes. Compiled runtime memory is also higher (10.3 vs 6.6 GB at batch 128). Next step: check the run logs for `skipping cudagraphs` or recompiles.
- **Why vLLM slows down at large batch.** A healthy H100 should finish the batch-128 shape in roughly 2 s, not 13 s. Because the extra cost scales with the number of requests rather than with compute, and is unaffected by CUDA graphs, it most likely comes from per-request CPU-side work in the engine in this vLLM 0.25 build or environment (candidates: the new V2 model runner, asynchronous scheduling). Ruled out: the harness, `min_tokens`, sampling mode, chunked prefill, tokenization/detokenization, and GPU throttling. Next step: rerun `vllm bench latency` at batch 128 with async scheduling off, the V1 model runner, and the torch profiler, and compare against an older vLLM release. Until then, treat vLLM results at batch ≥ 16 as specific to this environment.
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
  backend: transformers    # transformers | vllm (vllm: CUDA only, no int8/int4)
  precision: bf16          # fp32 | fp16 | bf16 | int8 | int4
  device: cuda             # cuda | mps
  batch_size: 16
  compile: true            # HF: static KV cache + CUDA-graph decode; vLLM: torch.compile + CUDA graphs

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
