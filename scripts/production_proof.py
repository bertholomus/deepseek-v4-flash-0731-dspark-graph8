#!/usr/bin/env python3
"""Fail-closed model, tool, strict-JSON, and exact-context proof."""
import argparse
import json
import os
import pathlib
import time
import urllib.request


def post(root, path, body, timeout=1200):
    request = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get(root, path, timeout=30):
    with urllib.request.urlopen(root + path, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("MODEL", "deepseek-v4-flash-0731"))
    parser.add_argument("--context-tokens", type=int, default=160000)
    parser.add_argument("--expected-max-model-len", type=int, default=1048576)
    parser.add_argument("--output", default="proof.json")
    args = parser.parse_args()
    root = args.base_url.rstrip("/")
    model = args.model
    result = {"schema_version": 1, "model_requested": model, "target_context_tokens": args.context_tokens, "started_at": time.time()}

    models = get(root, "/v1/models")["data"]
    primary = next(item for item in models if item.get("id") == model)
    result["model_gate"] = {"pass": primary.get("max_model_len") == args.expected_max_model_len, "id": primary.get("id"), "max_model_len": primary.get("max_model_len")}

    tools = [{"type": "function", "function": {"name": "calculator", "description": "Evaluate arithmetic", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}}}]
    user_message = "Use the calculator to compute 17 * 19. Do not calculate it yourself."
    first = post(root, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": user_message}], "tools": tools, "tool_choice": "auto", "temperature": 0, "max_tokens": 128})
    message = first["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    arguments = json.loads(calls[0]["function"]["arguments"]) if calls else {}
    result["tool_call"] = {"pass": bool(calls) and calls[0]["function"]["name"] == "calculator" and arguments == {"expression": "17 * 19"}, "name": calls[0]["function"]["name"] if calls else None, "arguments": arguments}

    continuation = post(root, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": user_message}, {"role": "assistant", "content": message.get("content"), "tool_calls": calls}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "323"}], "tools": tools, "temperature": 0, "max_tokens": 64})
    continuation_text = (continuation["choices"][0]["message"].get("content") or "").strip()
    result["tool_continuation"] = {"pass": "323" in continuation_text}

    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ready"]}, "nodes": {"type": "integer", "const": 2}}, "required": ["status", "nodes"], "additionalProperties": False}
    structured = post(root, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Return status as ready and nodes as 2."}], "temperature": 0, "max_tokens": 64, "response_format": {"type": "json_schema", "json_schema": {"name": "production_state", "strict": True, "schema": schema}}})
    state = json.loads(structured["choices"][0]["message"]["content"])
    result["strict_json"] = {"pass": state == {"status": "ready", "nodes": 2}, "object": state}

    marker = "GRAPH8_PUBLIC_CONTEXT_KEY_7C91"
    unit = "Distributed context verification filler block 0731. "
    prefix = f"The secret key is {marker}. Remember it exactly.\n"
    suffix = "\nReturn one JSON object with one property named key whose string value is the exact secret key from the beginning."

    def token_count(text):
        return post(root, "/tokenize", {"model": model, "prompt": text}, timeout=300)["count"]

    unit_tokens = token_count(unit)
    fixed_tokens = token_count(prefix + suffix)
    repetitions = max(1, (args.context_tokens - fixed_tokens) // unit_tokens)
    for _ in range(8):
        prompt = prefix + unit * repetitions + suffix
        measured = token_count(prompt)
        delta = args.context_tokens - measured
        if abs(delta) < 100:
            break
        repetitions = max(1, repetitions + int(delta / max(unit_tokens, 1)))
    prompt = prefix + unit * repetitions + suffix
    raw_tokens = token_count(prompt)
    retrieval_schema = {"type": "object", "properties": {"key": {"type": "string", "const": marker}}, "required": ["key"], "additionalProperties": False}
    started = time.time()
    retrieval = post(root, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 64, "response_format": {"type": "json_schema", "json_schema": {"name": "retrieved_key", "strict": True, "schema": retrieval_schema}}}, timeout=1800)
    retrieved = json.loads(retrieval["choices"][0]["message"]["content"])
    result["context"] = {"pass": retrieved == {"key": marker}, "raw_tokenized": raw_tokens, "usage": retrieval.get("usage"), "elapsed_s": time.time() - started}
    result["pass"] = all(result[name]["pass"] for name in ("model_gate", "tool_call", "tool_continuation", "strict_json", "context"))
    result["finished_at"] = time.time()
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
