# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed real-image prefill/capture workload for the local SenseSpec checkout.

Only tuning.json is an experimental input. The samples, reference features,
validation thresholds and measurement protocol remain fixed throughout a run.
The metric is median warm workload time divided by the original TP=4 time.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[2]
OUTPUT = ROOT / "outputs/loop-autoresearch-20261023"
MODEL = (
    "/mnt/afs_agents/yankuihao/model_ckpts/"
    "1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity():
    paths = [
        "h800conda-tmp.sh",
        "examples/qwen3.5-loop-mhc-image-text-dspark-1-node/config.yaml",
        "examples/qwen3.8-27b-image-text-1node/prefill.py",
        "sensespec/inference/engine/vllm_engine.py",
        "sensespec/inference/engine/mooncake_hidden_states_connector.py",
        "sensespec/training/data_fetcher.py",
        "outputs/loop-autoresearch-20261023/samples.jsonl",
    ]
    return {name: sha256(ROOT / name) for name in paths}


def snapshot(tensors):
    length = tensors["input_ids"].numel()
    indices = torch.linspace(0, length - 1, min(length, 257)).long()
    result = {
        name: tensors[name][indices].float().clone()
        for name in ("hidden_states", "last_hidden_states")
    }
    for name in ("input_ids", "loss_mask"):
        result[name] = tensors[name].clone()
    return result


def compare(actual, expected, label):
    failures = []
    diagnostics = {}
    for name in ("input_ids", "loss_mask"):
        if not torch.equal(actual[name], expected[name]):
            failures.append(f"{label}: {name} changed")
    for name in ("hidden_states", "last_hidden_states"):
        a, b = actual[name].double(), expected[name].double()
        relative_rmse = ((a - b).square().mean() / b.square().mean()).sqrt().item()
        cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        diagnostics[name] = {
            "relative_rmse": relative_rmse,
            "cosine": cosine.item(),
            "max_abs": (a - b).abs().max().item(),
            "bitwise_equal": torch.equal(a, b),
        }
        if not (relative_rmse <= 0.02 and cosine >= 0.999):
            failures.append(f"{label}: {name} numerical mismatch: {diagnostics[name]}")
    return failures, diagnostics


def run(smoke=False):
    from sensespec.config.mooncake_config import MooncakeConfig
    from sensespec.config.train_config import config_to_flat_args, load_config
    from sensespec.data.dataset import load_conversation_dataset
    from sensespec.transfer.mooncake import EagleMooncakeStore, MooncakeMaster

    spec = importlib.util.spec_from_file_location(
        "capture_validation",
        ROOT / "examples/qwen3.8-27b-image-text-1node/prefill.py",
    )
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    tuning = json.loads((BENCH / "tuning.json").read_text())
    if tuning["tp_size"] not in (1, 2, 4):
        raise ValueError("This experiment reserves four GPUs for training")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    observed_identity = identity()
    reference_path = OUTPUT / "reference.json"
    reference = (
        json.loads(reference_path.read_text()) if reference_path.exists() else None
    )
    if reference and reference["identity"] != observed_identity:
        raise RuntimeError("Frozen input or SenseSpec code changed since baseline")
    args = config_to_flat_args(
        load_config(
            str(ROOT / "examples/qwen3.5-loop-mhc-image-text-dspark-1-node/config.yaml")
        )
    )
    args.target_model_path = MODEL
    args.train_data_path = str(OUTPUT / "samples.jsonl")
    args.output_dir = str(OUTPUT / "engine")
    args.cache_dir = str(OUTPUT / "tokenization")
    args.num_proc = 1
    args.shuffle_dataset = False
    args.complete_data_pass = True
    args.supervision_manifest_path = str(
        ROOT / "cache/qwen3.5-loop-mhc-image-text-dspark-1-node/full-data/manifest.json"
    )
    args.inference_num_gpus = tuning["tp_size"]
    args.inference_num_gpus_per_engine = tuning["tp_size"]
    args.vllm_tp_size = tuning["tp_size"]
    for key in ("max_num_batched_tokens", "disable_custom_all_reduce"):
        args.vllm_extra_args[key] = tuning[key]
    prompts = load_conversation_dataset(args)
    source_rows = {
        row["data_id"]: row
        for row in map(json.loads, (OUTPUT / "samples.jsonl").read_text().splitlines())
    }
    prompts = sorted(prompts, key=lambda p: list(source_rows).index(p["data_id"]))
    if smoke:
        prompts = prompts[:1]
    assert len(prompts) == (1 if smoke else 4)
    tokenizer = validation.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    master = MooncakeMaster()
    info = master.start(master.find_free_port(51135), master.find_free_port(8763))
    host = info["master_addr"].split(":")[0]
    config = MooncakeConfig(
        local_hostname=host,
        master_server_address=info["master_addr"],
        metadata_server=f"http://{host}:{info['metadata_port']}/metadata",
        global_segment_size="16GB",
        local_buffer_size="8GB",
        host_buffer_size="8GB",
        max_seq_len=131072,
        hidden_dim=4096,
        num_aux_layers=5,
    )
    engine = validation.VllmEngine(args, rank=0, num_gpus_per_engine=tuning["tp_size"])
    store = EagleMooncakeStore(config)
    observations, rounds, failures = [], [], []
    baseline_snapshots = {}
    started = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        engine.init(config)
        store.setup(device="cpu")
        assert engine.aux_hidden_state_layer_ids == [4, 28, 52, 76, 79]
        for round_index in range(1 if smoke else 4):
            times = []
            for prompt in prompts:
                did = prompt["data_id"]
                window = source_rows[did]["metadata"]["supervision_window"]
                begin = time.perf_counter()
                results = engine.generate(
                    [did],
                    [prompt["input_ids"]],
                    packed_loss_mask_list=[prompt.get("packed_loss_mask")],
                    multimodal_inputs=[prompt.get("multimodal_inputs")],
                    supervision_windows=[window],
                )
                assert len(results) == 1
                result = results[0]
                tensors = validation.read_capture(store, result)
                seconds = time.perf_counter() - begin
                times.append(seconds)
                length = result["seq_len"]
                assert length == window["expanded_tokens"]
                assert len(result["prefill_completion_token_ids"]) == 1
                assert tensors["hidden_states"].shape == (length, 20480)
                assert tensors["last_hidden_states"].shape == (length, 4096)
                assert tensors["input_ids"].tolist() == result["input_ids_list"]
                for name in ("hidden_states", "last_hidden_states"):
                    assert torch.isfinite(tensors[name]).all(), name
                # Full supervision checks and numerical comparisons are untimed.
                validation.validate_supervision_capture(
                    tokenizer,
                    tensors["input_ids"],
                    tensors["loss_mask"],
                    source_rows[did],
                )
                snap = snapshot(tensors)
                snap["completion"] = result["prefill_completion_token_ids"]
                if reference:
                    expected = torch.load(
                        OUTPUT / "reference-captures" / f"{did}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                elif did in baseline_snapshots:
                    expected = baseline_snapshots[did]
                else:
                    baseline_snapshots[did] = snap
                    expected = snap
                mismatches, diagnostics = compare(snap, expected, did)
                failures.extend(mismatches)
                if snap["completion"] != expected["completion"]:
                    failures.append(f"{did}: greedy completion changed")
                observation = {
                    "round": round_index,
                    "warmup": round_index == 0,
                    "data_id": did,
                    "seconds": seconds,
                    "tokens": length,
                    "images": window["image_count"],
                    "completion": snap["completion"],
                    "numerical": diagnostics,
                }
                observations.append(observation)
                print(json.dumps(observation), flush=True)
                del tensors, snap, expected
                gc.collect()
            if round_index > 0:
                rounds.append(sum(times))
    finally:
        engine.shutdown()
        store.close()
        master.shutdown()
    elapsed = time.time() - started
    if smoke:
        print(json.dumps({"smoke_seconds": elapsed, "failures": failures}), flush=True)
        return
    seconds = statistics.median(rounds)
    if reference is None:
        if failures:
            raise RuntimeError(f"Baseline is not repeatable: {failures}")
        capture_dir = OUTPUT / "reference-captures"
        capture_dir.mkdir(exist_ok=True)
        for did, snap in baseline_snapshots.items():
            torch.save(snap, capture_dir / f"{did}.pt")
        reference = {"identity": observed_identity, "median_seconds": seconds}
        with reference_path.open("x") as handle:
            handle.write(json.dumps(reference, indent=2) + "\n")
    report = {
        "tuning": tuning,
        "identity": observed_identity,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "vllm"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "observations": observations,
        "round_seconds": rounds,
        "median_seconds": seconds,
        "baseline_seconds": reference["median_seconds"],
        "time_ratio": seconds / reference["median_seconds"],
        "correctness_failures": failures,
        "wall_seconds": elapsed,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    (OUTPUT / f"measurement-{stamp}.json").write_text(rendered)
    (OUTPUT / "latest.json").write_text(rendered)
    print(json.dumps({"time_ratio": report["time_ratio"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    run(smoke=parser.parse_args().smoke)
