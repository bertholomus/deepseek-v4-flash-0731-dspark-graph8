#!/usr/bin/env python3
"""Warm streaming TTFT, decode-rate, and DSpark acceptance benchmark."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import time
import urllib.request

CASES = [
    {"name": "count300", "prompt": "Count from 1 to 300, separated by commas. Return only the numbers and commas.", "max_tokens": 700},
    {"name": "dataclasses20", "prompt": "Write valid Python defining exactly 20 dataclasses named Record01 through Record20. Each has fields id: int, name: str, active: bool. Return only one Python code block and end it with the comment # END_DATACLASSES.", "max_tokens": 1400},
    {"name": "prose500", "prompt": "Write a coherent 500-word technical explanation of failure isolation in a two-node tensor-parallel inference service. Use plain prose, no headings or bullets, and finish with the exact marker END_PROSE.", "max_tokens": 1100},
]


def request_json(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def metrics(root):
    text = urllib.request.urlopen(root + "/metrics", timeout=10).read().decode()
    out = {}
    for name in ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_accepted_tokens_total"):
        matches = re.findall(r"^" + re.escape(name) + r"\{[^\n]*\}\s+([0-9.eE+-]+)$", text, re.M)
        out[name] = sum(float(value) for value in matches)
    return out


def validate(name, content):
    if name == "count300":
        return content.strip() == ",".join(map(str, range(1, 301)))
    if name == "dataclasses20":
        return all(f"class Record{i:02d}:" in content for i in range(1, 21)) and content.rstrip("`\n ").endswith("# END_DATACLASSES")
    if name == "prose500":
        return content.rstrip().endswith("END_PROSE")
    return False


def stream_one(root, model, case):
    body = {"model": model, "messages": [{"role": "user", "content": case["prompt"]}], "temperature": 0, "max_tokens": case["max_tokens"], "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(root + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    pieces = []
    finish_reason = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece and first is None:
                    first = time.perf_counter()
                pieces.append(piece)
                finish_reason = choices[0].get("finish_reason") or finish_reason
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    content = "".join(pieces)
    tokens = (usage or {}).get("completion_tokens", 0)
    decode_elapsed = max(0.001, finished - (first or finished))
    return {"ttft_s": (first or finished) - started, "elapsed_s": finished - started, "completion_tokens": tokens, "decode_tok_s": tokens / decode_elapsed, "finish_reason": finish_reason, "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "output_valid": validate(case["name"], content), "usage": usage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-v4-flash-0731"))
    parser.add_argument("--label", default="graph8-reproduction")
    parser.add_argument("--output", default="benchmark.json")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    root = args.base_url.rstrip("/")
    report = {"schema_version": 1, "label": args.label, "model": args.model, "repetitions": args.repetitions, "cases": []}
    for case in CASES:
        stream_one(root, args.model, case)
        before = metrics(root)
        runs = [stream_one(root, args.model, case) for _ in range(args.repetitions)]
        after = metrics(root)
        draft = after["vllm:spec_decode_num_draft_tokens_total"] - before["vllm:spec_decode_num_draft_tokens_total"]
        accepted = after["vllm:spec_decode_num_accepted_tokens_total"] - before["vllm:spec_decode_num_accepted_tokens_total"]
        entry = {"name": case["name"], "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in runs), "median_ttft_s": statistics.median(r["ttft_s"] for r in runs), "draft_tokens_delta": draft, "accepted_tokens_delta": accepted, "acceptance_pct": 100 * accepted / draft if draft else None, "all_outputs_valid": all(r["output_valid"] for r in runs), "runs": runs}
        report["cases"].append(entry)
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: value for key, value in entry.items() if key != "runs"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
