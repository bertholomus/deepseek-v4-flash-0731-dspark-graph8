# GB10 stability and headroom profile

## Why the defaults changed

The initial two-node production profile used GPU memory utilization `0.80`,
8,192 batched tokens, and Graph-8. A node later disappeared abruptly while the
distributed endpoint went down. The prior boot journal ended without an
orderly shutdown, so the initiating cause cannot be proven conclusively.
However, fresh launches of that profile emitted `NV_ERR_NO_MEMORY` allocation
failures on both GB10 nodes and retained only about 13–15 GiB `MemAvailable`.
That made insufficient host/runtime headroom a proven material risk even
though it does not prove that OOM was the sole initiating cause.

## Recommended balanced profile

```text
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=4096
MAX_CUDAGRAPH_CAPTURE_SIZE=8
GPU_MEMORY_UTILIZATION=0.75
MTP_NUM_TOKENS=5
UCX_MEM_MMAP_HOOK_MODE=none
UCX_RCACHE_MAX_UNRELEASED=1024
```

This keeps the Graph-8 speed benefit and two ordinary concurrent sessions while
restoring host headroom. The observed KV capacity was 1,372,579 tokens, or 1.31
times the configured one-request context ceiling. It is a shared pool, not a
promise of two simultaneous full-1M-token prompts.

## Measured concurrent proof

Two simultaneous requests each generated an exact 1,005-token bounded sequence:

- `running=2`, `waiting=0` throughout generation;
- 21.05 seconds per request;
- 47.74 tok/s per stream;
- 95.48 tok/s aggregate;
- approximately 20 GiB and 18 GiB `MemAvailable` after the proof;
- memory PSI `avg10=0.00`;
- zero new NVIDIA allocation errors after the healthy-start baseline;
- zero service restarts.

## Startup allocation-probe rule

GB10/vLLM startup profiling can itself emit allocation-probe
`NV_ERR_NO_MEMORY` events before the endpoint becomes ready. Do not ignore
those events, but do not let a watchdog repeatedly kill every otherwise
progressing cold start either.

Use this sequence:

1. Preserve the pre-start NVIDIA-error counters.
2. Start under supervision while monitoring forward progress, SSH latency,
   `MemAvailable`, PSI, Xid/reset/hung-task evidence, and endpoint readiness.
3. If the endpoint becomes healthy and the hosts retain the required headroom,
   record a new post-start baseline.
4. Treat any later counter increase as runtime evidence and combine it with
   memory, PSI, responsiveness, endpoint progress, Xid/reset, hung-task, or OOM
   evidence before deciding to stop.
5. Fail closed immediately on combined severe evidence or a disappearing node.

An allocation-error counter by itself is evidence, not a complete diagnosis.
The baseline must never be advanced while the service is stalled or unhealthy.

## Promotion gates for more aggressive settings

Before increasing sequence count, batched tokens, context, graph capture, or
GPU memory utilization, require:

1. Exact rendered launch-argument verification on both ranks.
2. Healthy model identity and configured context from `/v1/models`.
3. Matched single-stream and target-concurrency benchmarks.
4. `running=N`, `waiting=0` during the intended concurrency proof.
5. Long-context retrieval, native tool call, continuation, and strict JSON.
6. Stable `MemAvailable`, swap, and PSI with responsive SSH/control paths.
7. No post-baseline NVIDIA allocation growth, Xid/reset, OOM-kill, or hung task.
8. Coordinated cold restart followed by the same functional and pressure gates.
9. A preserved rollback profile and a watchdog that writes its safety latch
   before attempting a distributed stop.

Do not present a launch, open socket, or one short completion as stability proof.