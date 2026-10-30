# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare a Loop operation with a frozen prior implementation."""

import argparse
import importlib.util
import json
import statistics
import sys
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.layers import sensenova_loop_mhc as candidate
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument(
        "--operation", choices=("residual_merge", "normalize"), default="residual_merge"
    )
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.sweep and args.operation != "residual_merge":
        parser.error("--sweep only supports residual_merge")
    spec = importlib.util.spec_from_file_location("baseline_mhc", args.baseline_source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    torch.manual_seed(73)
    with torch.inference_mode():
        tokens = args.tokens
        residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
        branch = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        coefficients = torch.randn(tokens, 24, device="cuda")
        post = 2 * coefficients[:, 4:8].sigmoid()
        mix = coefficients[:, 8:].reshape(tokens, 4, 4).softmax(-1)
        inputs = (
            (residual,)
            if args.operation == "normalize"
            else (residual, branch, post, mix)
        )
        operations = {
            name: partial(getattr(module, args.operation), *inputs, backend="triton")
            for name, module in (("baseline", baseline), ("candidate", candidate))
        }
        reference = operations["baseline"]()
        if args.sweep:
            for batch in (1, 2, 4):
                for block in (64, 128, 256, 512):
                    output = torch.empty_like(residual)

                    def launch(block=block, batch=batch, output=output):
                        candidate._merge_tensor_core_kernel[
                            (triton.cdiv(4096, block), triton.cdiv(tokens, batch))
                        ](
                            residual,
                            branch,
                            post,
                            mix,
                            output,
                            tokens,
                            4096,
                            post.stride(0),
                            mix.stride(0),
                            block,
                            batch,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )

                    launch()
                    equal = torch.equal(
                        output.view(torch.uint8), reference.view(torch.uint8)
                    )
                    ms = (
                        triton.testing.do_bench(launch, warmup=30, rep=100)
                        if equal
                        else None
                    )
                    print(
                        json.dumps(
                            {
                                "batch": batch,
                                "block": block,
                                "bitwise_equal": equal,
                                "milliseconds": ms,
                            }
                        ),
                        flush=True,
                    )
            return
        actual = operations["candidate"]()
        assert torch.equal(reference.view(torch.uint8), actual.view(torch.uint8))
        times = {name: [] for name in operations}
        for name in (
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
        ):
            ms = triton.testing.do_bench(
                operations[name], warmup=50, rep=200, quantiles=[0.5, 0.2, 0.8]
            )[0]
            times[name].append(ms)
            print(json.dumps({"name": name, "milliseconds": ms}), flush=True)
        medians = {name: statistics.median(values) for name, values in times.items()}
        print(
            json.dumps(
                {
                    "operation": args.operation,
                    "tokens": tokens,
                    "bitwise_equal": True,
                    "median_ms": medians,
                    "speedup": medians["baseline"] / medians["candidate"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
