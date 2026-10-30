# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Loop's log-domain router boundary with its original PyTorch path."""

from collections import defaultdict

import pytest
import torch

from vllm.model_executor.layers.sensenova_loop_mhc import log_sinkhorn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("tokens", [1, 7, 16, 8191, 8192])
@pytest.mark.parametrize("scale", [0.0, 1e-7, 1.0, 100.0])
def test_fused_sinkhorn_preserves_original_fp32_output(tokens, scale, streams):
    generator = torch.Generator(device="cuda").manual_seed(17)
    # Match the noncontiguous slice produced by the router's projection.
    logits = (
        torch.randn(tokens, streams * (streams + 2), device="cuda", generator=generator)
        * scale
    )[:, 2 * streams :].view(tokens, streams, streams)

    def reference():
        value = logits
        for _ in range(20):
            value = value - torch.logsumexp(value, dim=-1, keepdim=True)
            value = value - torch.logsumexp(value, dim=-2, keepdim=True)
        return value.exp()

    expected = reference()
    assert torch.equal(expected, reference()), "reference is not repeatable"
    actual = log_sinkhorn(logits, 20, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("streams,iterations", [(3, 20), (4, 2)])
def test_unsupported_router_shapes_keep_reference_and_gradients(streams, iterations):
    logits = torch.randn(2, streams, streams, requires_grad=True)
    expected = log_sinkhorn(logits, iterations)
    actual = log_sinkhorn(logits, iterations, backend="triton")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("tokens", [1, 7, 31, 32, 33, 8191, 8192])
@pytest.mark.parametrize("scale", [0.0, 1e-7, 1.0, 100.0])
def test_mhc_fusions_preserve_bitwise_outputs_and_input(tokens, scale, streams):
    from vllm.model_executor.layers import sensenova_loop_mhc as mhc

    generator = torch.Generator(device="cuda").manual_seed(73)
    residual = (
        torch.randn(
            tokens,
            streams,
            4096,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * scale
    )
    original = residual.clone()
    branch = (
        torch.randn(
            tokens, 4096, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * scale
    )
    coefficients = torch.randn(
        tokens, streams * (streams + 2), device="cuda", generator=generator
    )
    pre = coefficients[:, :streams]
    pre.copy_(pre.sigmoid())
    post = coefficients[:, streams : 2 * streams]
    post.copy_(2 * post.sigmoid())
    mix = coefficients[:, 2 * streams :].reshape(tokens, streams, streams).softmax(-1)
    with torch.inference_mode():
        for operation, inputs in (
            (mhc.normalize, (residual,)),
            (mhc.pre_aggregate, (pre, residual)),
            (mhc.residual_merge, (residual, branch, post, mix)),
        ):
            expected = operation(*inputs)
            repeated = operation(*inputs)
            actual = operation(*inputs, backend="triton")
            assert torch.equal(expected.view(torch.uint8), repeated.view(torch.uint8))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(residual.view(torch.uint8), original.view(torch.uint8))


@pytest.mark.parametrize("grad", [False, True])
def test_mhc_fallback_preserves_outputs_and_gradients(grad):
    from vllm.model_executor.layers import sensenova_loop_mhc as mhc

    residual = torch.randn(2, 3, 8, requires_grad=grad)
    pre = torch.randn(2, 3, requires_grad=grad)
    branch = torch.randn(2, 8, requires_grad=grad)
    post = torch.randn(2, 3, requires_grad=grad)
    mix = torch.randn(2, 3, 3, requires_grad=grad)
    for operation, inputs in (
        (mhc.normalize, (residual,)),
        (mhc.pre_aggregate, (pre, residual)),
        (mhc.residual_merge, (residual, branch, post, mix)),
    ):
        actual = operation(*inputs, backend="triton")
        torch.testing.assert_close(actual, operation(*inputs), rtol=0, atol=0)
        if grad:
            actual.sum().backward()
    if grad:
        for value in (residual, pre, branch, post, mix):
            assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_noncontiguous_residual_uses_exact_reference_path():
    from vllm.model_executor.layers import sensenova_loop_mhc as mhc

    residual = torch.randn(7, 4, 32, device="cuda", dtype=torch.bfloat16)[..., ::2]
    pre = torch.rand(7, 4, device="cuda")
    branch = torch.randn(7, 16, device="cuda", dtype=torch.bfloat16)
    mix = torch.rand(7, 4, 4, device="cuda")
    for operation, inputs in (
        (mhc.normalize, (residual,)),
        (mhc.pre_aggregate, (pre, residual)),
        (mhc.residual_merge, (residual, branch, pre, mix)),
    ):
        expected = operation(*inputs)
        actual = operation(*inputs, backend="triton")
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden", [64, 4095, 4096])
@pytest.mark.parametrize("transpose_mix", [False, True])
@pytest.mark.parametrize("mix_dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_merge_preserves_reference_for_coefficient_types_and_layouts(
    hidden, transpose_mix, mix_dtype
):
    from vllm.model_executor.layers import sensenova_loop_mhc as mhc

    generator = torch.Generator(device="cuda").manual_seed(73)
    residual = torch.randn(
        33, 4, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    branch = residual[:, 0].contiguous()
    post = torch.rand(33, 4, device="cuda", generator=generator)
    mix = torch.rand(33, 4, 4, device="cuda", generator=generator).to(mix_dtype)
    if transpose_mix:
        mix = mix.transpose(1, 2)
    expected = mhc.residual_merge(residual, branch, post, mix)
    actual = mhc.residual_merge(residual, branch, post, mix, backend="triton")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_variable_prefill_lengths_reuse_mhc_kernels(monkeypatch):
    """A new prefill tail must not compile another set of mHC kernels."""
    from triton import knobs

    from vllm.model_executor.layers import sensenova_loop_mhc as mhc

    # Isolate the in-process cache from kernels warmed by other tests.
    for name in (
        "_cast_square_kernel",
        "_normalize_scale_kernel",
        "_sinkhorn_kernel",
        "_pre_aggregate_kernel",
        "_merge_epilogue_kernel",
        "_merge_tensor_core_kernel",
    ):
        kernel = getattr(mhc, name)
        monkeypatch.setattr(kernel, "device_caches", defaultdict(kernel.create_binder))

    generator = torch.Generator(device="cuda").manual_seed(73)

    def run(tokens):
        residual = torch.randn(
            tokens, 4, 4096, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        branch = torch.randn(
            tokens, 4096, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        coefficients = torch.randn(tokens, 24, device="cuda", generator=generator)
        pre = coefficients[:, :4].sigmoid()
        post = 2 * coefficients[:, 4:8].sigmoid()
        # Match the router's layout, including its distinct row stride at N=1.
        log_mix = coefficients[:, 8:].unflatten(-1, (4, 4))
        mix = mhc.log_sinkhorn(log_mix, 20)
        for operation, inputs in (
            (mhc.log_sinkhorn, (log_mix, 20)),
            (mhc.normalize, (residual,)),
            (mhc.pre_aggregate, (pre, residual)),
            (mhc.residual_merge, (residual, branch, post, mix)),
        ):
            expected = operation(*inputs)
            actual = operation(*inputs, backend="triton")
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    cache_misses = []

    def on_cache_miss(**kwargs):
        fn = kwargs["fn"]
        if fn.module == mhc.__name__:
            cache_misses.append(fn.name)

    with torch.inference_mode():
        run(8192)
        # Router strides are legitimately specialized; warm both layouts.
        run(1)
        monkeypatch.setattr(knobs.runtime, "jit_cache_hook", on_cache_miss)
        # Include single-token, aligned, unaligned and observed production tails.
        for tokens in (1, 7, 16, 1334, 3903, 8191, 8193):
            run(tokens)
        torch.accelerator.synchronize()

    assert not cache_misses, f"Variable lengths recompiled mHC: {cache_misses}"
