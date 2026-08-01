# Methodology and interpretation

## Hardware and serving shape

Measurements were collected from a two-node NVIDIA GB10-class tensor-parallel service. Each node held a complete, independently verified copy of the official DeepSeek-V4-Flash-0731 checkpoint. The service used TP2 over RoCE v2, DSpark K5, NVFP4 DS-MLA KV cache, FlashInfer B12X MoE, and ordinary CUDA Graphs with observed capture sizes `[1, 2, 4, 8]`.

## Decode throughput

`decode_tok_s` is completion tokens divided by elapsed time after receipt of the first non-empty streamed content token. It is not end-to-end request throughput and excludes TTFT. Each reported value is the median of three measured runs after one unmeasured warm-up for that prompt shape.

## TTFT

TTFT is wall-clock time from request dispatch until the first non-empty streamed content token. The reported 0.186–0.249 second values are warm, short-prompt, single-request measurements. They do not predict cold-start, long-prefill, loaded-server, or wide-area-network latency.

## DSpark acceptance

Acceptance is the change in vLLM's accepted speculative-token counter divided by the change in its drafted-token counter over the measured repetitions. High acceptance explains the 84–92 tok/s structured results. The prose shape had 32.10% acceptance and 40.08 tok/s, demonstrating why structured throughput must not be advertised as a universal rate.

## Context evidence

Production exact retrieval used 159,979 API-reported prompt tokens, cached tokens 0, and returned the exact bounded key after 97.50 seconds. An equivalent pre-promotion Graph-8 qualification pair completed an uncached 949,896-prompt-token exact retrieval with a 1,057.98 second TTFT. The latter supports the 1M configuration but was not rerun on the final production pair because it takes roughly 18 minutes and the promoted runtime/checkpoint profile was the same.

## Graph-4 versus Graph-8

Matched A/B measurements on the qualification pair showed Graph-8 gains of 26.42% for count-to-300, 14.78% for dataclasses, and 27.24% for prose. The K5 proposer produces a six-token verification shape; a capture set ending at 4 misses it, while explicit capture through 8 includes it.

## Exactness and parity

Hard exactness gates passed for bounded output, retrieval keys, native tool name/arguments, tool-result continuation, and closed-world strict JSON. One unconstrained prose comparison produced 256 tokens with DSpark and 255 without speculation. The orchestration harness therefore returned nonzero for byte parity even though the text was semantically comparable. This package does not claim byte-identical unconstrained sampling.

## Reproduction controls

Record at minimum:

- model repository and exact revision;
- 48 shard hashes on each node;
- runtime image tag, registry digest, and local image ID;
- vLLM/FlashInfer runtime versions;
- TP topology and fabric interfaces;
- node-specific RoCEv2 GID indices;
- exact environment and launch command;
- CUDA Graph capture set observed in startup logs;
- warm-up count, repetitions, request concurrency, prompt, and sampling settings;
- raw API usage and vLLM speculative counters.

Do not compare these numbers to another system unless tokenization, prompt, completion length, concurrency, warm-up, and throughput definitions match.
