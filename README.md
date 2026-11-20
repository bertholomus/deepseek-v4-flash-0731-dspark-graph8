---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
library_name: vllm
tags:
  - deepseek
  - dspark
  - speculative-decoding
  - vllm
  - dgx-spark
  - gb10
  - tensor-parallel
  - long-context
  - nvfp4
---

# DeepSeek-V4-Flash-0731 DSpark Graph-8 on two GB10 nodes

> **Archived.** This DeepSeek-V4-Flash-0731 (vLLM / DSpark) recipe is superseded by [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10). This repository is read-only.

A reproducible deployment and benchmark package for the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
checkpoint on two NVIDIA GB10-class nodes using vLLM tensor parallelism, DSpark speculative decoding, NVFP4 DS-MLA KV cache, FlashInfer B12X MoE, RoCE, and ordinary CUDA Graph capture through batch shape 8.

> **Independent community recipe.** This repository is not an official DeepSeek release and contains no model weights. Download the official checkpoint from DeepSeek-AI. DSpark accelerates target-verified generation; it does not change the target model's knowledge or semantics.

## Headline result

### Stabilized balanced profile

The current safer production profile runs **two concurrent sequences** with
`GPU_MEMORY_UTILIZATION=0.75`, 4,096 batched tokens, Graph-8, and the UCX
registration-cache safeguards documented in [`docs/STABILITY.md`](docs/STABILITY.md).
In a simultaneous two-request proof, both requests stayed scheduled
(`running=2`, `waiting=0`) and each generated 1,005 exact bounded tokens in
21.05 seconds:

| Concurrency | Per-stream decode | Aggregate decode | Post-baseline allocation errors |
|---:|---:|---:|---:|
| 2 | **47.74 tok/s** | **95.48 tok/s** | **0 on both nodes** |

At the end of that proof the nodes retained approximately 20 GiB and 18 GiB
`MemAvailable`, memory PSI `avg10` was 0.00, and the service had zero restarts.
This is the recommended default profile; it does not imply two simultaneous
full-1M-token requests because the KV pool is shared.

### Initial warm single-request qualification

On a two-node production pair, three warm single-request repetitions per shape produced:

| Workload | Median decode | Median TTFT | DSpark acceptance |
|---|---:|---:|---:|
| Count 1–300, exact bounded output | **91.89 tok/s** | **0.186 s** | **98.81%** |
| Twenty Python dataclasses | **84.43 tok/s** | **0.249 s** | **98.66%** |
| Low-predictability technical prose | **40.08 tok/s** | **0.235 s** | **32.10%** |

These medians came from the initial Graph-8 qualification profile before the
headroom hardening above. The replaced GGUF/llama.cpp lane delivered about
9.3 tok/s on the same two-node class, so the measured uplift was approximately
9.9×, 9.1×, and 4.3× respectively. These are workload-specific measurements,
not a universal throughput floor, and should not be compared directly with the
two-stream aggregate measurement.

Graph-8 improved matched decode over Graph-4 by:

| Workload | Graph-4 | Graph-8 | Gain |
|---|---:|---:|---:|
| Count 1–300 | 74.28 | 93.90 tok/s | +26.42% |
| Dataclasses | 75.22 | 86.33 tok/s | +14.78% |
| Technical prose | 32.12 | 40.87 tok/s | +27.24% |

The K5 proposer verifies six-token shapes. Capturing `[1,2,4,8]` rather than stopping at 4 covers that shape.

## Qualification gates

- Official checkpoint revision: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- 48/48 safetensor shards verified independently on both ranks
- 96/96 draft parameters loaded; no skipped tensors
- Configured context: **1,048,576 tokens**
- Stabilized-profile KV capacity observed at startup: **1,372,579 tokens** (1.31× the configured full-context ceiling, shared across active requests)
- Stabilized two-request proof: **47.74 tok/s per stream**, **95.48 tok/s aggregate**, `running=2`, `waiting=0`
- Stabilized post-proof headroom: approximately **20 GiB / 18 GiB MemAvailable**, memory PSI `avg10=0.00`
- Production exact uncached retrieval: **159,979 prompt tokens**, 97.50 s, cached tokens 0
- Equivalent pre-promotion qualification pair: **949,896 prompt tokens**, exact key retrieval, 1,057.98 s TTFT, cached tokens 0
- Native OpenAI-format tool call: passed
- Tool-result continuation: passed
- Closed-world strict JSON schema: passed
- Production service restart and post-restart smoke: passed

One unconstrained prose parity sample differed by one output token between speculative and non-speculative runs (256 vs. 255 tokens). Bounded outputs, retrieval keys, tool names/arguments, and strict JSON remained exact. Do not interpret this package as a claim of byte-identical unconstrained prose.

## Canonical runtime profile

```text
Model: DeepSeek-V4-Flash-0731
Tensor parallel: 2 nodes
DSpark draft tokens: 5 (K5)
KV cache: nvfp4_ds_mla
MoE backend: flashinfer_b12x
VLLM_USE_BREAKABLE_CUDAGRAPH=0
CUDA graph capture set observed: [1, 2, 4, 8]
--max-cudagraph-capture-size 8
--max-model-len 1048576
--max-num-seqs 2
--max-num-batched-tokens 4096
--gpu-memory-utilization 0.75
--block-size 256
UCX_MEM_MMAP_HOOK_MODE=none
UCX_RCACHE_MAX_UNRELEASED=1024
RoCE v2 / NCCL IB
```

Pinned runtime image:

```text
ghcr.io/anemll/dspark-vllm-gx10:0.1.1
registry digest: sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8
validated local image ID: sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8
```

## Quick start

### 1. Prepare both nodes

Each node needs:

- one GB10-class CUDA GPU;
- Docker with Compose and NVIDIA Container Toolkit;
- RoCE-capable connectivity visible under `/dev/infiniband`;
- the complete official 0731 checkpoint at the same local path;
- passwordless SSH only if you add your own coordinator wrapper.

Verify all 48 model shards against a trusted manifest before launch.

### 2. Configure node-specific environments

Copy `runtime/.env.example` to `.env.worker` and `.env.head`. Set a different `VLLM_HOST_IP` and `NCCL_IB_GID_INDEX` on each node. `MASTER_ADDR` must be the head node's fabric address on both ranks.

Do not reuse one GID index across nodes without checking sysfs: RoCEv2 GID indices can differ and can move after reboot or link changes.

Do not restore the earlier `0.80` GPU-utilization / 8,192-batched-token defaults
without a fresh pressure-qualified experiment. Read [`docs/STABILITY.md`](docs/STABILITY.md)
before changing memory, sequence, graph, or watchdog settings.

### 3. Start worker, then head

Worker rank:

```bash
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 \
  docker compose --env-file runtime/.env.worker \
  -f runtime/docker-compose.yml -p deepseek-v4-flash-0731 up -d
```

Head rank:

```bash
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= \
  docker compose --env-file runtime/.env.head \
  -f runtime/docker-compose.yml -p deepseek-v4-flash-0731 up -d
```

Keep the API loopback-only unless you deliberately provide authentication and firewalling. If remote clients need access, put an authenticated reverse proxy in front of vLLM.

### 4. Verify

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=deepseek-v4-flash-0731 \
  python3 scripts/production_proof.py --context-tokens 160000

BASE_URL=http://127.0.0.1:8000 MODEL=deepseek-v4-flash-0731 \
  python3 scripts/benchmark.py --repetitions 3 --output benchmark.json
```

## Repository contents

- `runtime/docker-compose.yml` — two-node vLLM/DSpark Graph-8 profile
- `runtime/.env.example` — node-specific configuration template
- `scripts/production_proof.py` — model, tools, strict JSON, and long-context gates
- `scripts/benchmark.py` — streaming TTFT/decode/acceptance harness
- `benchmarks/production-summary.json` — sanitized measured results
- `docs/METHODOLOGY.md` — definitions, test conditions, caveats, and parity disclosure
- `docs/STABILITY.md` — OOM/headroom incident, stabilized defaults, startup-baseline rule, and promotion gates
- `NOTICE` — upstream projects and pinned revisions

## Attribution

This package builds on the official DeepSeek model and community GB10 work from Anemll, MiaAI-Lab, Tony Deangelo, and Weschera. Exact repositories and commits are recorded in `NOTICE`.

## License

Code and documentation in this package are MIT-licensed. The official model and weights are separately provided by DeepSeek-AI under MIT; obtain them from the upstream repository. Preserve all upstream notices when redistributing upstream code.
