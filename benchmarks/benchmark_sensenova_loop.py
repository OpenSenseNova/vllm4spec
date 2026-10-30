# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-request, exact-length Loop TTFT through streaming completions."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx


def worker_rpc(client: httpx.Client, url: str, method: str, *args: Any) -> list[Any]:
    response = client.post(
        url.rstrip("/") + "/collective_rpc", json={"method": method, "args": args}
    )
    response.raise_for_status()
    return response.json()["results"]


def executed_backend(worker: dict[str, Any], backend: str, fusion: str) -> bool:
    from vllm.model_executor.layers.sensenova_loop_mhc import FUSION_MODES

    selected = FUSION_MODES[fusion]
    counts = worker["sinkhorn_calls"]
    if (
        counts[backend] <= 0
        or counts["torch" if backend == "triton" else "triton"] != 0
        or worker["fusion_modes"] != [fusion]
    ):
        return False
    for name in FUSION_MODES["all"]:
        expected = "triton" if name in selected else "torch"
        counts = worker["fusion_calls"][name]
        if (
            counts[expected] <= 0
            or counts["torch" if expected == "triton" else "triton"] != 0
        ):
            return False
    return True


def audit_outputs(
    client: httpx.Client,
    url: str,
    model: str,
    prompt: list[int],
    *,
    capture_layers: list[int],
    tp_size: int,
    reference_backend: str = "torch",
    candidate_backend: str = "triton",
    reference_fusion: str = "none",
    candidate_fusion: str = "none",
    chunk_tokens: int = 8192,
) -> dict[str, Any]:
    observations = []
    try:
        for backend, fusion in (
            (reference_backend, reference_fusion),
            (reference_backend, reference_fusion),
            (candidate_backend, candidate_fusion),
        ):
            worker_rpc(client, url, "loop_benchmark_backend", backend, fusion)
            worker_rpc(client, url, "loop_benchmark_capture_start")
            try:
                result = measure_request(client, url, model, prompt)
            finally:
                captures = worker_rpc(client, url, "loop_benchmark_capture_stop")
            observations.append(
                {
                    "backend": backend,
                    "fusion": fusion,
                    "token_ids": result["token_ids"],
                    "request": result,
                    "ranks": captures,
                    "workers": worker_rpc(client, url, "loop_benchmark_status"),
                }
            )
    finally:
        worker_rpc(
            client, url, "loop_benchmark_backend", reference_backend, reference_fusion
        )
    reference, repeated, candidate = observations
    required_states = {f"layer_{index}" for index in capture_layers} | {
        "final_post_norm"
    }
    expected_rows = [
        min(chunk_tokens, len(prompt) - offset)
        for offset in range(0, len(prompt), chunk_tokens)
    ]
    expected_positions = [
        {
            "tokens": rows,
            "start": offset,
            "end": offset + rows - 1,
            "contiguous": True,
        }
        for offset, rows in zip(range(0, len(prompt), chunk_tokens), expected_rows)
    ]
    populated = all(
        len(observation["ranks"]) == tp_size
        and all(
            set(rank) == required_states | {"logits", "positions"}
            and rank["positions"] == expected_positions
            and all(
                [value["shape"][0] for value in rank[name]] == expected_rows
                for name in required_states
            )
            and all(
                value["finite"]
                for name in required_states | {"logits"}
                for value in rank[name]
            )
            for rank in observation["ranks"]
        )
        and any(rank["logits"] for rank in observation["ranks"])
        for observation in observations
    )
    executed = all(
        len(observation["workers"]) == tp_size
        and all(
            executed_backend(worker, observation["backend"], observation["fusion"])
            for worker in observation["workers"]
        )
        for observation in observations
    )
    return {
        "observations": observations,
        "reference_repeatable": populated
        and reference["ranks"] == repeated["ranks"]
        and reference["token_ids"] == repeated["token_ids"],
        "candidate_equal": populated
        and executed
        and reference["ranks"] == candidate["ranks"]
        and reference["token_ids"] == candidate["token_ids"],
    }


def source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        "vllm/model_executor/models/sensenova_loop.py",
        "vllm/model_executor/layers/sensenova_loop_mhc.py",
        "benchmarks/benchmark_sensenova_loop.py",
        "benchmarks/sensenova_loop_benchmark_worker.py",
    ]
    git_dir = root / ".git"
    if git_dir.is_file():
        git_dir = root / git_dir.read_text().strip().removeprefix("gitdir: ")
    commit = subprocess.check_output(
        ["git", f"--git-dir={git_dir}", f"--work-tree={root}", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return {
        "git_head": commit,
        "sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in paths
        },
    }


def validate_config(
    config: dict[str, Any], max_model_len: int = 131072, chunk_tokens: int = 8192
) -> None:
    expected: dict[str, dict[str, Any]] = {
        "model_config": {
            "max_model_len": max_model_len,
            "enforce_eager": True,
            "dtype": "bfloat16",
            "quantization": None,
        },
        "parallel_config": {
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
        },
        "scheduler_config": {
            "max_num_batched_tokens": chunk_tokens,
            "max_num_seqs": 1,
            "enable_chunked_prefill": True,
        },
        "cache_config": {
            "enable_prefix_caching": False,
            "mamba_ssm_cache_dtype": "float32",
        },
    }
    if config["parallel_config"]["tensor_parallel_size"] not in (4, 8):
        raise ValueError("Benchmark requires TP=4 or TP=8")
    for section, values in expected.items():
        for key, value in values.items():
            actual = config[section][key]
            if str(actual).removeprefix("torch.") != str(value):
                raise ValueError(
                    f"Benchmark requires {section}.{key}={value}, got {actual}"
                )
    if (
        config.get("speculative_config") is not None
        or config.get("profiler_config", {}).get("profiler") is not None
    ):
        raise ValueError("Benchmark requires no speculative decoder or engine profiler")


def engine_metrics(client: httpx.Client, url: str) -> dict[str, float]:
    response = client.get(url.rstrip("/") + "/metrics")
    response.raise_for_status()
    names = ("vllm:time_to_first_token_seconds", "vllm:request_prefill_time_seconds")
    values = {}
    for name in names:
        for suffix in ("_sum", "_count"):
            key = name + suffix
            values[key] = sum(
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith(key + "{")
            )
    return values


def measure_request(
    client: httpx.Client, url: str, model: str, prompt: list[int]
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "logprobs": 1,
    }
    first_token = None
    tokens: list[int] = []
    usage = None
    finish_reason = None
    done = False
    started = time.perf_counter()
    with client.stream(
        "POST", url.rstrip("/") + "/v1/completions", json=payload
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"Generation failed: {event['error']}")
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids and first_token is None:
                    first_token = time.perf_counter() - started
                tokens.extend(ids)
                finish_reason = choice.get("finish_reason") or finish_reason
            usage = event.get("usage") or usage
    elapsed = time.perf_counter() - started
    if (
        not done
        or first_token is None
        or len(tokens) != 1
        or finish_reason != "length"
        or usage is None
        or usage["prompt_tokens"] != len(prompt)
        or usage["completion_tokens"] != 1
    ):
        raise RuntimeError(
            f"Incomplete exact-length request: done={done}, tokens={tokens}, "
            f"finish_reason={finish_reason}, usage={usage}"
        )
    return {
        "ttft_seconds": first_token,
        "request_seconds": elapsed,
        "input_tokens": len(prompt),
        "output_tokens": len(tokens),
        "token_ids": tokens,
        "finish_reason": finish_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", required=True, help="Served checkpoint/tokenizer path"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=131071)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--backend", choices=["torch", "triton"], default="torch")
    parser.add_argument(
        "--reference-backend", choices=["torch", "triton"], default="torch"
    )
    for option in ("--fusion", "--reference-fusion"):
        parser.add_argument(
            option, choices=["none", "norm", "pre", "merge", "all"], default="none"
        )
    parser.add_argument("--profile-chunks", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile each selected chunk in a separate request after timing",
    )
    args = parser.parse_args()
    if (
        args.repetitions < 5
        or not 0 < args.prompt_tokens < args.max_model_len <= 262144
        or args.chunk_tokens <= 0
        or any(
            index < 0 or index * args.chunk_tokens >= args.prompt_tokens
            for index in args.profile_chunks
        )
    ):
        parser.error("Require >=5 repeats, valid context length and prefill chunks")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed_tokens = tokenizer.encode("请用一句话介绍你自己。", add_special_tokens=False)
    prompt = (
        seed_tokens * ((args.prompt_tokens + len(seed_tokens) - 1) // len(seed_tokens))
    )[: args.prompt_tokens]
    report: dict[str, Any] = {
        "model": args.model,
        "url": args.url,
        "prompt_tokens": len(prompt),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        "measurement": "client POST to first SSE choice containing generated token IDs",
        "sampling": {"temperature": 0, "seed": 0, "max_tokens": 1, "ignore_eos": True},
        "warmup": None,
        "runs": [],
        "backend": args.backend,
        "reference_backend": args.reference_backend,
        "fusion": args.fusion,
        "reference_fusion": args.reference_fusion,
        "chunk_tokens": args.chunk_tokens,
        "client_source": source_identity(),
        "checkpoint_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(Path(args.model).glob("*"))
            if path.is_file()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=1800, trust_env=False) as client:
        metadata = client.get(args.url.rstrip("/") + "/server_info?config_format=json")
        metadata.raise_for_status()
        report["effective_config"] = metadata.json()["vllm_config"]
        report["server_system"] = metadata.json()["system_env"]
        report["server_runner"] = metadata.json()["vllm_env"].get(
            "VLLM_USE_V2_MODEL_RUNNER"
        )
        validate_config(
            report["effective_config"], args.max_model_len, args.chunk_tokens
        )
        report["startup_workers"] = worker_rpc(
            client, args.url, "loop_benchmark_status"
        )
        tp_size = report["effective_config"]["parallel_config"]["tensor_parallel_size"]
        if len(report["startup_workers"]) != tp_size:
            raise RuntimeError(f"Expected {tp_size} TP ranks")
        capture_layers = report["startup_workers"][0]["capture_layers"]
        if not capture_layers or any(
            worker["capture_layers"] != capture_layers
            for worker in report["startup_workers"]
        ):
            raise RuntimeError("Workers disagree on the Loop capture layers")
        report["capture_layers"] = capture_layers
        for worker in report["startup_workers"]:
            for path, digest in worker["loaded_source_sha256"].items():
                if report["client_source"]["sha256"].get(path) != digest:
                    raise RuntimeError(
                        f"Server loaded different source for {path}; restart it"
                    )
        if args.backend == "triton" or args.fusion != "none":
            if any(worker["streams"] != [4] for worker in report["startup_workers"]):
                raise ValueError(
                    "Triton mHC benchmarks require four residual streams; "
                    "use --backend torch --fusion none for this checkpoint"
                )
            report["numerical_audit"] = audit_outputs(
                client,
                args.url,
                args.model,
                prompt,
                capture_layers=capture_layers,
                tp_size=tp_size,
                reference_backend=args.reference_backend,
                candidate_backend=args.backend,
                reference_fusion=args.reference_fusion,
                candidate_fusion=args.fusion,
                chunk_tokens=args.chunk_tokens,
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not (
                report["numerical_audit"]["reference_repeatable"]
                and report["numerical_audit"]["candidate_equal"]
            ):
                raise RuntimeError(
                    "Numerical audit failed; workers restored to reference"
                )
        worker_rpc(
            client, args.url, "loop_benchmark_backend", args.backend, args.fusion
        )
        report["warmup"] = measure_request(client, args.url, args.model, prompt)
        for _ in range(args.repetitions):
            before = engine_metrics(client, args.url)
            result = measure_request(client, args.url, args.model, prompt)
            after = engine_metrics(client, args.url)
            result["engine_metrics"] = {
                name: after[name + "_sum"] - before[name + "_sum"]
                if after[name + "_count"] - before[name + "_count"] == 1
                else None
                for name in (
                    "vllm:time_to_first_token_seconds",
                    "vllm:request_prefill_time_seconds",
                )
            }
            report["runs"].append(result)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        report["short_request_after_long"] = measure_request(
            client, args.url, args.model, seed_tokens
        )
        report["measured_workers"] = worker_rpc(
            client, args.url, "loop_benchmark_status"
        )
        if any(
            not executed_backend(worker, args.backend, args.fusion)
            for worker in report["measured_workers"]
        ):
            worker_rpc(
                client,
                args.url,
                "loop_benchmark_backend",
                args.reference_backend,
                args.reference_fusion,
            )
            raise RuntimeError("Candidate fell back to torch; no fused timing accepted")
        if args.profile:
            report["profiles"] = []
            for chunk_index in args.profile_chunks:
                worker_rpc(
                    client, args.url, "loop_benchmark_profile_start", chunk_index
                )
                try:
                    request = measure_request(client, args.url, args.model, prompt)
                finally:
                    ranks = worker_rpc(client, args.url, "loop_benchmark_profile_stop")
                report["profiles"].append(
                    {"chunk_index": chunk_index, "request": request, "ranks": ranks}
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    times = [run["ttft_seconds"] for run in report["runs"]]
    report["summary"] = {
        "median_seconds": statistics.median(times),
        "min_seconds": min(times),
        "max_seconds": max(times),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
