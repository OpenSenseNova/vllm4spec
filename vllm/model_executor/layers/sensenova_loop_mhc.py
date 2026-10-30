# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova Loop mHC operations with explicit FP32/BF16 rounding boundaries."""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton

logger = init_logger(__name__)
sinkhorn_execution_counts = {"torch": 0, "triton": 0}
FUSION_MODES = {
    "none": frozenset(),
    "norm": frozenset({"normalize"}),
    "pre": frozenset({"pre_aggregate"}),
    "merge": frozenset({"residual_merge"}),
    "all": frozenset({"normalize", "pre_aggregate", "residual_merge"}),
}
fusion_execution_counts = {
    name: {"torch": 0, "triton": 0} for name in FUSION_MODES["all"]
}


def _supported(residual: torch.Tensor, *inputs: torch.Tensor) -> bool:
    return (
        residual.is_cuda
        and torch.version.hip is None
        and residual.dtype == torch.bfloat16
        and residual.ndim == 3
        and residual.shape[1] == 4
        and 0 < residual.numel() < 2**31
        and residual.is_contiguous()
        and all(
            value.device == residual.device and not value.requires_grad
            for value in (residual, *inputs)
        )
    )


def _use_fusion(name: str, backend: str, supported: bool) -> bool:
    if backend not in ("torch", "triton"):
        raise ValueError(f"Unknown SenseNova Loop mHC backend: {backend}")
    fused = backend == "triton" and supported
    fusion_execution_counts[name]["triton" if fused else "torch"] += 1
    if backend == "triton":
        if fused:
            logger.info_once("SenseNova Loop mHC %s executing triton", name)
        else:
            logger.warning_once("SenseNova Loop mHC %s falling back to torch", name)
    return fused


# Prefill tails vary per request. Keep lengths out of the specialization key,
# including Triton's single-value/alignment specialization for runtime integers.
@triton.jit(do_not_specialize=["SIZE"])
def _cast_square_kernel(X, Y, SIZE, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, offset < SIZE, other=0).to(tl.float32)
    tl.store(Y + offset, value * value, offset < SIZE)


@triton.jit(do_not_specialize=["SIZE"])
def _normalize_scale_kernel(
    X, SCALE, Y, SIZE, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, offset < SIZE, other=0).to(tl.float32)
    scale = tl.load(SCALE + offset // WIDTH, offset < SIZE, other=0)
    tl.store(Y + offset, value * scale, offset < SIZE)


@triton.jit(do_not_specialize=["N"])
def _pre_aggregate_kernel(
    X, PRE, Y, N, D: tl.constexpr, PS: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = offset // D, offset % D
    acc = tl.full((BLOCK,), 0, tl.float32)
    for stream in range(4):
        value = tl.load(X + token * 4 * D + stream * D + col, token < N, other=0).to(
            tl.float32
        )
        pre = (
            tl.load(PRE + token * PS + stream, token < N, other=0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        acc = tl.fma(value, pre, acc)
    tl.store(Y + offset, acc, token < N)


@triton.jit(do_not_specialize=["N"])
def _merge_epilogue_kernel(
    MIXED,
    BRANCH,
    POST,
    Y,
    N,
    D: tl.constexpr,
    PS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, stream, col = offset // (4 * D), (offset // D) % 4, offset % D
    mixed = tl.load(MIXED + offset, token < N, other=0).to(tl.float32)
    branch = tl.load(BRANCH + token * D + col, token < N, other=0).to(tl.float32)
    post = (
        tl.load(POST + token * PS + stream, token < N, other=0)
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    # Preserve the materialized BF16 product before the separate BF16 add.
    scaled = (post * branch).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offset, mixed + scaled, token < N)


@triton.jit(do_not_specialize=["N"])
def _merge_tensor_core_kernel(
    X,
    BRANCH,
    POST,
    MIX,
    Y,
    N,
    D: tl.constexpr,
    PS: tl.constexpr,
    MS: tl.constexpr,
    BLOCK: tl.constexpr,
    BATCH: tl.constexpr,
):
    token = tl.program_id(1) * BATCH
    row = tl.arange(0, 16)
    inner = tl.arange(0, 32)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_token = token + row // 4
    inner_token = token + inner // 4
    # Pack independent 4x4 matrices along the diagonal of one MMA tile.
    # Tensor Core accumulation preserves the original BF16 GEMM rounding.
    mix = tl.load(
        MIX + row_token[:, None] * MS + (row[:, None] % 4) * 4 + inner[None, :] % 4,
        (row[:, None] < 4 * BATCH)
        & (row_token[:, None] < N)
        & (row[:, None] // 4 == inner[None, :] // 4),
        other=0,
    ).to(tl.bfloat16)
    residual = tl.load(
        X + token * 4 * D + inner[:, None] * D + col[None, :],
        (inner[:, None] < 4 * BATCH) & (inner_token[:, None] < N) & (col[None, :] < D),
        other=0,
    )
    mixed = tl.dot(mix, residual).to(tl.bfloat16).to(tl.float32)
    branch = tl.load(
        BRANCH + row_token[:, None] * D + col[None, :],
        (row[:, None] < 4 * BATCH) & (row_token[:, None] < N) & (col[None, :] < D),
        other=0,
    ).to(tl.float32)
    post = tl.load(
        POST + row_token * PS + row % 4,
        (row < 4 * BATCH) & (row_token < N),
        other=0,
    )
    post = post.to(tl.bfloat16).to(tl.float32)
    # The GEMM and product each round to BF16 before the final addition.
    scaled = (post[:, None] * branch).to(tl.bfloat16).to(tl.float32)
    tl.store(
        Y + row_token[:, None] * 4 * D + (row[:, None] % 4) * D + col[None, :],
        mixed + scaled,
        (row[:, None] < 4 * BATCH) & (row_token[:, None] < N) & (col[None, :] < D),
    )


def normalize(residual: torch.Tensor, *, backend: str = "torch") -> torch.Tensor:
    if _use_fusion("normalize", backend, backend == "triton" and _supported(residual)):
        tokens, streams, hidden = residual.shape
        squared = torch.empty(
            (tokens, streams * hidden), dtype=torch.float32, device=residual.device
        )
        large_prefill = tokens >= 32 and streams * hidden == 16384
        _cast_square_kernel[(triton.cdiv(residual.numel(), 1024),)](
            residual,
            squared,
            residual.numel(),
            1024,
            num_warps=8 if large_prefill else 4,
            enable_fp_fusion=False,
        )
        # Retain PyTorch's reduction and rsqrt to preserve their FP32 results.
        scale = torch.rsqrt(squared.mean(-1, keepdim=True) + 1e-6)
        # The same stream has consumed squared; reuse its FP32 storage.
        output = squared
        block = 4096 if large_prefill else 1024
        _normalize_scale_kernel[(triton.cdiv(residual.numel(), block),)](
            residual,
            scale,
            output,
            residual.numel(),
            streams * hidden,
            block,
            enable_fp_fusion=False,
        )
        return output
    flat = residual.flatten(-2).float()
    return flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)


def pre_aggregate(
    pre: torch.Tensor, residual: torch.Tensor, *, backend: str = "torch"
) -> torch.Tensor:
    supported = (
        backend == "triton"
        and _supported(residual, pre)
        and pre.dtype == torch.float32
        and pre.shape == residual.shape[:2]
        and pre.stride(-1) == 1
    )
    if _use_fusion("pre_aggregate", backend, supported):
        tokens, _, hidden = residual.shape
        output = torch.empty(
            (tokens, hidden), device=residual.device, dtype=residual.dtype
        )
        _pre_aggregate_kernel[(triton.cdiv(tokens * hidden, 1024),)](
            residual,
            pre,
            output,
            tokens,
            hidden,
            pre.stride(0),
            1024,
            enable_fp_fusion=False,
        )
        return output
    return torch.matmul(pre.to(residual.dtype).unsqueeze(-2), residual).squeeze(-2)


def residual_merge(
    residual: torch.Tensor,
    branch: torch.Tensor,
    post: torch.Tensor,
    mix: torch.Tensor,
    *,
    backend: str = "torch",
) -> torch.Tensor:
    supported = (
        backend == "triton"
        and _supported(residual, branch, post, mix)
        and branch.dtype == torch.bfloat16
        and branch.shape == (residual.shape[0], residual.shape[-1])
        and branch.is_contiguous()
        and post.dtype == torch.float32
        and post.shape == residual.shape[:2]
        and post.stride(-1) == 1
        and mix.shape == (residual.shape[0], 4, 4)
    )
    if _use_fusion("residual_merge", backend, supported):
        tokens, _, hidden = residual.shape
        output = torch.empty_like(residual)
        if (
            tokens >= 32
            and hidden == 4096
            and mix.dtype == torch.float32
            and mix.stride(1) == 4
            and mix.stride(2) == 1
        ):
            _merge_tensor_core_kernel[
                (triton.cdiv(hidden, 256), triton.cdiv(tokens, 4))
            ](
                residual,
                branch,
                post,
                mix,
                output,
                tokens,
                hidden,
                post.stride(0),
                mix.stride(0),
                256,
                4,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return output
        mixed = torch.matmul(mix.to(residual.dtype), residual)
        _merge_epilogue_kernel[(triton.cdiv(residual.numel(), 1024),)](
            mixed,
            branch,
            post,
            output,
            tokens,
            hidden,
            post.stride(0),
            1024,
            enable_fp_fusion=False,
        )
        return output
    return torch.matmul(mix.to(residual.dtype), residual) + post.to(
        branch.dtype
    ).unsqueeze(-1) * branch.unsqueeze(-2)


def expand_streams(hidden: torch.Tensor, streams: int) -> torch.Tensor:
    return hidden.unsqueeze(-2).expand(-1, streams, -1).contiguous()


def collapse_streams(residual: torch.Tensor) -> torch.Tensor:
    return residual.mean(dim=-2)


@triton.jit(do_not_specialize=["N"])
def _sinkhorn_kernel(
    X,
    Y,
    N,
    STRIDE: tl.constexpr,
    ITERATIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.arange(0, 4)
    col = tl.arange(0, 4)
    offsets = (
        token[:, None, None] * STRIDE + row[None, :, None] * 4 + col[None, None, :]
    )
    value = tl.load(X + offsets, token[:, None, None] < N, other=0)
    for _ in range(ITERATIONS):
        maximum = tl.max(value, 2)
        exp = tldevice.exp(value - maximum[:, :, None])
        a = tl.sum(tl.where(col[None, None, :] == 0, exp, 0), 2)
        b = tl.sum(tl.where(col[None, None, :] == 1, exp, 0), 2)
        c = tl.sum(tl.where(col[None, None, :] == 2, exp, 0), 2)
        d = tl.sum(tl.where(col[None, None, :] == 3, exp, 0), 2)
        # Match PyTorch 2.13 CUDA's fast-dimension reduction order explicitly.
        total = (a + c) + (b + d)
        value = value - (tldevice.log(total) + maximum)[:, :, None]
        maximum = tl.max(value, 1)
        exp = tldevice.exp(value - maximum[:, None, :])
        # PyTorch's strided four-element reduction accumulates in index order.
        a = tl.sum(tl.where(row[None, :, None] == 0, exp, 0), 1)
        b = tl.sum(tl.where(row[None, :, None] == 1, exp, 0), 1)
        c = tl.sum(tl.where(row[None, :, None] == 2, exp, 0), 1)
        d = tl.sum(tl.where(row[None, :, None] == 3, exp, 0), 1)
        total = ((a + b) + c) + d
        value = value - (tldevice.log(total) + maximum)[:, None, :]
    output_offsets = (
        token[:, None, None] * 16 + row[None, :, None] * 4 + col[None, None, :]
    )
    tl.store(Y + output_offsets, tldevice.exp(value), token[:, None, None] < N)


def log_sinkhorn(
    log_mix: torch.Tensor, iterations: int, *, backend: str = "torch"
) -> torch.Tensor:
    """Normalize rows then columns, preserving all FP32 iterations."""
    if backend not in ("torch", "triton"):
        raise ValueError(f"Unknown SenseNova Loop Sinkhorn backend: {backend}")
    if (
        backend == "triton"
        and log_mix.is_cuda
        and torch.version.hip is None
        and log_mix.dtype == torch.float32
        and log_mix.ndim == 3
        and log_mix.shape[-2:] == (4, 4)
        and log_mix.stride()[-2:] == (4, 1)
        and not log_mix.requires_grad
        and iterations == 20
        and log_mix.shape[0] > 0
    ):
        output = torch.empty_like(log_mix, memory_format=torch.contiguous_format)
        _sinkhorn_kernel[(triton.cdiv(log_mix.shape[0], 16),)](
            log_mix,
            output,
            log_mix.shape[0],
            log_mix.stride(0),
            iterations,
            16,
            enable_fp_fusion=False,
        )
        logger.info_once(
            "SenseNova Loop mHC Sinkhorn executing triton (FP32, 20 iterations)"
        )
        sinkhorn_execution_counts["triton"] += 1
        return output
    if backend == "triton":
        logger.warning_once(
            "SenseNova Loop mHC Sinkhorn falling back to torch: unsupported input"
        )
    sinkhorn_execution_counts["torch"] += 1
    for _ in range(iterations):
        log_mix = log_mix - torch.logsumexp(log_mix, dim=-1, keepdim=True)
        log_mix = log_mix - torch.logsumexp(log_mix, dim=-2, keepdim=True)
    return log_mix.exp()
