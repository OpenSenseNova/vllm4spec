# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure Loop mHC primitives after checking their original rounding results."""

import argparse
import json
from functools import partial

import torch

from vllm.model_executor.layers import sensenova_loop_mhc as mhc
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def sweep_launches(residual, pre, log_mix):
    """Measure launch geometry without changing arithmetic or reductions."""
    tokens, streams, hidden = residual.shape
    size = residual.numel()
    squared = residual.flatten(-2).float().square()
    scale = torch.rsqrt(squared.mean(-1, keepdim=True) + 1e-6)
    normalized = mhc.normalize(residual)
    aggregated = mhc.pre_aggregate(pre, residual)
    mixed = mhc.log_sinkhorn(log_mix, 20)
    square_output = torch.empty_like(squared)
    norm_output = torch.empty_like(normalized)
    pre_output = torch.empty_like(aggregated)
    mix_output = torch.empty_like(mixed)
    cases = (
        (
            "cast_square",
            mhc._cast_square_kernel,
            (residual, square_output, size),
            size,
            square_output,
            squared,
        ),
        (
            "normalize_scale",
            mhc._normalize_scale_kernel,
            (residual, scale, norm_output, size, streams * hidden),
            size,
            norm_output,
            normalized,
        ),
        (
            "pre_aggregate",
            mhc._pre_aggregate_kernel,
            (residual, pre, pre_output, tokens, hidden, pre.stride(0)),
            tokens * hidden,
            pre_output,
            aggregated,
        ),
        (
            "sinkhorn",
            mhc._sinkhorn_kernel,
            (log_mix, mix_output, tokens, log_mix.stride(0), 20),
            tokens,
            mix_output,
            mixed,
        ),
    )
    for name, kernel, inputs, length, output, reference in cases:
        blocks = (
            (4, 8, 16, 32, 64, 128)
            if name == "sinkhorn"
            else (256, 512, 1024, 2048, 4096, 8192)
        )
        for block in blocks:
            for warps in (2, 4, 8):
                launch = partial(
                    kernel[(triton.cdiv(length, block),)],
                    *inputs,
                    block,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
                launch()
                equal = torch.equal(
                    output.view(torch.uint8), reference.view(torch.uint8)
                )
                ms = (
                    triton.testing.do_bench(
                        launch, warmup=30, rep=100, quantiles=[0.5, 0.2, 0.8]
                    )[0]
                    if equal
                    else None
                )
                print(
                    json.dumps(
                        {
                            "operation": name,
                            "tokens": tokens,
                            "block": block,
                            "num_warps": warps,
                            "bitwise_equal": equal,
                            "milliseconds": ms,
                        }
                    ),
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--sweep-launches", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("tokens must be positive")
    torch.manual_seed(73)
    tokens = args.tokens
    with torch.inference_mode():
        residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
        branch = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        coefficients = torch.randn(tokens, 24, device="cuda")
        pre = coefficients[:, :4].sigmoid()
        post = 2 * coefficients[:, 4:8].sigmoid()
        mix = coefficients[:, 8:].reshape(tokens, 4, 4).softmax(-1)
        if args.sweep_launches:
            sweep_launches(residual, pre, coefficients[:, 8:].reshape(tokens, 4, 4))
            return
        for operation, inputs in (
            (mhc.normalize, (residual,)),
            (mhc.pre_aggregate, (pre, residual)),
            (mhc.residual_merge, (residual, branch, post, mix)),
        ):
            reference = operation(*inputs)
            candidate = operation(*inputs, backend="triton")
            if not torch.equal(
                reference.view(torch.uint8), candidate.view(torch.uint8)
            ):
                raise RuntimeError(f"Numerical mismatch in {operation.__name__}")
            del reference, candidate
            times = {
                backend: triton.testing.do_bench(
                    partial(operation, *inputs, backend=backend),
                    warmup=50,
                    rep=200,
                    quantiles=[0.5, 0.2, 0.8],
                )
                for backend in ("torch", "triton")
            }
            print(
                json.dumps(
                    {
                        "operation": operation.__name__,
                        "tokens": tokens,
                        "device": current_platform.get_device_name(),
                        "torch": str(torch.__version__),
                        "triton": str(triton.__version__),
                        "bitwise_equal": True,
                        "milliseconds_p50_p20_p80": times,
                        "speedup": times["torch"][0] / times["triton"][0],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
