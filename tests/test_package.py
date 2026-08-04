#!/usr/bin/env python3
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_summary_contract(self):
        data = json.loads((ROOT / "benchmarks/production-summary.json").read_text())
        self.assertEqual(data["model"]["revision"], "9e165c30e2704aec5d9d593cce3eebd58bbef1cb")
        self.assertEqual(data["runtime"]["max_cudagraph_capture_size"], 8)
        self.assertEqual(data["runtime"]["mtp_num_tokens"], 5)
        self.assertEqual(data["runtime"]["max_num_seqs"], 2)
        self.assertEqual(data["runtime"]["max_num_batched_tokens"], 4096)
        self.assertEqual(data["runtime"]["gpu_memory_utilization"], 0.75)
        self.assertEqual(data["balanced_concurrency_proof"]["aggregate_tokens_per_second"], 95.48)
        self.assertEqual(len(data["production_three_run_medians"]), 3)
        self.assertTrue(data["production_gates"]["strict_json"])

    def test_stabilized_runtime_defaults(self):
        env = (ROOT / "runtime/.env.example").read_text()
        compose = (ROOT / "runtime/docker-compose.yml").read_text()
        for expected in [
            "MAX_NUM_SEQS=2",
            "MAX_NUM_BATCHED_TOKENS=4096",
            "MAX_CUDAGRAPH_CAPTURE_SIZE=8",
            "GPU_MEMORY_UTILIZATION=0.75",
            "UCX_MEM_MMAP_HOOK_MODE=none",
            "UCX_RCACHE_MAX_UNRELEASED=1024",
        ]:
            self.assertIn(expected, env)
        self.assertIn("${MAX_NUM_SEQS:-2}", compose)
        self.assertIn("${MAX_NUM_BATCHED_TOKENS:-4096}", compose)
        self.assertIn("${GPU_MEMORY_UTILIZATION:-0.75}", compose)
        self.assertIn("${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}", compose)
        self.assertIn("UCX_MEM_MMAP_HOOK_MODE", compose)
        self.assertIn("UCX_RCACHE_MAX_UNRELEASED", compose)

    def test_no_private_infrastructure_addresses(self):
        files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix != ".pyc"]
        text = "\n".join(path.read_text(errors="ignore") for path in files)
        self.assertIsNone(re.search(r"\b100\.(?:\d{1,3}\.){2}\d{1,3}\b", text))
        self.assertNotIn("bertholomus" + "@", text)

    def test_no_common_secret_shapes(self):
        files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix != ".pyc"]
        text = "\n".join(path.read_text(errors="ignore") for path in files)
        forbidden = [r"hf_[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"sk-[A-Za-z0-9]{20,}"]
        for pattern in forbidden:
            self.assertIsNone(re.search(pattern, text), pattern)

    def test_readme_declares_no_weights(self):
        text = (ROOT / "README.md").read_text()
        self.assertIn("contains no model weights", text)
        self.assertIn("not an official DeepSeek release", text)


if __name__ == "__main__":
    unittest.main()
