# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional worker extension for local Loop benchmarks, never loaded by default."""

import hashlib
import inspect
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from vllm.model_executor.layers import sensenova_loop_mhc as mhc_ops
from vllm.model_executor.layers.sensenova_loop_mhc import (
    log_sinkhorn,
    sinkhorn_execution_counts,
)
from vllm.model_executor.models.sensenova_loop import (
    SenseNovaLoopDecoderLayer,
    SenseNovaLoopMHCRouter,
    SenseNovaLoopModel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

_LOADED_SOURCE = {
    logical_path: hashlib.sha256(Path(loaded_path).read_bytes()).hexdigest()
    for logical_path, loaded_path in (
        (
            "vllm/model_executor/models/sensenova_loop.py",
            inspect.getfile(SenseNovaLoopModel),
        ),
        (
            "vllm/model_executor/layers/sensenova_loop_mhc.py",
            inspect.getfile(log_sinkhorn),
        ),
        ("benchmarks/sensenova_loop_benchmark_worker.py", __file__),
    )
}


def position_summary(module, inputs, kwargs) -> dict[str, Any]:
    positions = (
        inspect.signature(module.forward).bind(*inputs, **kwargs).arguments["positions"]
    )
    positions = positions.detach().cpu()
    axes_equal = True
    if positions.ndim == 2 and positions.shape[0] > 0:
        axes_equal = torch.equal(positions, positions[0:1].expand_as(positions))
        positions = positions[0]
    if positions.ndim != 1 or positions.numel() == 0:
        return {"tokens": 0, "start": None, "end": None, "contiguous": False}
    start, end = int(positions[0]), int(positions[-1])
    return {
        "tokens": positions.numel(),
        "start": start,
        "end": end,
        "contiguous": axes_equal
        and torch.equal(positions, torch.arange(start, end + 1)),
    }


def capture_layer_ids(loop: SenseNovaLoopModel) -> list[int]:
    """Capture Loop boundaries and the last two pre-final layer boundaries."""
    indices = set(range(max(0, len(loop.layers) - 3), len(loop.layers) - 1))
    for index, layer in enumerate(loop.layers):
        if layer.expand_streams and index > 0:
            indices.add(index - 1)
        if layer.collapse_streams:
            indices.add(index)
    return sorted(i for i in indices if i + 1 in loop.capture_boundary_slots)


class LoopBenchmarkWorker:
    # vLLM mixes this extension into its existing worker class.
    def loop_benchmark_status(self) -> dict[str, Any]:
        model = self.get_model()  # type: ignore[attr-defined]
        loop = next(m for m in model.modules() if isinstance(m, SenseNovaLoopModel))
        routers = [m for m in model.modules() if isinstance(m, SenseNovaLoopMHCRouter)]
        torch.accelerator.synchronize()
        matmul = torch.backends.cuda.matmul
        return {
            "device": current_platform.get_device_name(),
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
            "cuda": torch.version.cuda,
            "loaded_source_sha256": _LOADED_SOURCE,
            "capture_layers": capture_layer_ids(loop),
            "sinkhorn_calls": dict(sinkhorn_execution_counts),
            "fusion_calls": {
                name: dict(counts)
                for name, counts in mhc_ops.fusion_execution_counts.items()
            },
            "fusion_modes": sorted(
                {
                    m.mhc_fusion
                    for m in model.modules()
                    if isinstance(
                        m, (SenseNovaLoopMHCRouter, SenseNovaLoopDecoderLayer)
                    )
                }
            ),
            "routers": len(routers),
            "backends": sorted({m.sinkhorn_backend for m in routers}),
            "router_dtypes": sorted({str(m.weight.dtype) for m in routers}),
            "streams": sorted({m.streams for m in routers}),
            "iterations": sorted({m.iterations for m in routers}),
            "tf32": matmul.allow_tf32,
            "bf16_reduced_precision_reduction": (
                matmul.allow_bf16_reduced_precision_reduction
            ),
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
        }

    def loop_benchmark_backend(self, backend: str, fusion: str = "none") -> None:
        if backend not in ("torch", "triton") or fusion not in mhc_ops.FUSION_MODES:
            raise ValueError((backend, fusion))
        model = self.get_model()  # type: ignore[attr-defined]
        for module in model.modules():
            if isinstance(module, SenseNovaLoopMHCRouter):
                module.sinkhorn_backend = backend
            if isinstance(module, (SenseNovaLoopMHCRouter, SenseNovaLoopDecoderLayer)):
                module.mhc_fusion = fusion
        sinkhorn_execution_counts.update(torch=0, triton=0)
        for counts in mhc_ops.fusion_execution_counts.values():
            counts.update(torch=0, triton=0)
        torch.accelerator.reset_peak_memory_stats()

    def loop_benchmark_capture_start(self) -> None:
        model = self.get_model()  # type: ignore[attr-defined]
        loop = next(m for m in model.modules() if isinstance(m, SenseNovaLoopModel))
        self._loop_hashes: dict[str, list[dict[str, Any]]] = {}
        self._loop_handles = []

        self._loop_hashes["positions"] = []

        def capture_positions(module, inputs, kwargs):
            self._loop_hashes["positions"].append(
                position_summary(module, inputs, kwargs)
            )

        self._loop_handles.append(
            loop.register_forward_pre_hook(capture_positions, with_kwargs=True)
        )

        def capture(name):
            self._loop_hashes[name] = []

            def hook(module, inputs, output):
                if output is None:
                    return
                raw = output.detach().contiguous().view(torch.uint8).cpu().numpy()
                self._loop_hashes[name].append(
                    {
                        "shape": list(output.shape),
                        "dtype": str(output.dtype),
                        "finite": bool(torch.isfinite(output).all()),
                        "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                    }
                )

            return hook

        for index in capture_layer_ids(loop):
            self._loop_handles.append(
                loop.layers[index].register_forward_hook(capture(f"layer_{index}"))
            )
        self._loop_handles.append(
            loop.norm.register_forward_hook(capture("final_post_norm"))
        )
        language_model = (
            model.language_model if hasattr(model, "language_model") else model
        )
        self._loop_handles.append(
            language_model.logits_processor.register_forward_hook(capture("logits"))
        )

    def loop_benchmark_capture_stop(self) -> dict[str, Any]:
        for handle in self._loop_handles:
            handle.remove()
        self._loop_handles = []
        return self._loop_hashes

    def loop_benchmark_profile_start(self, chunk_index: int = 0) -> None:
        """Profile one selected chunk, then restore functions and remove hooks."""
        if chunk_index < 0:
            raise ValueError("chunk_index must be nonnegative")
        model = self.get_model()  # type: ignore[attr-defined]
        loop = next(m for m in model.modules() if isinstance(m, SenseNovaLoopModel))
        self._loop_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        current_chunk = -1
        self._loop_profile_chunk = None
        self._loop_profile_active = False
        self._loop_profile_handles = []
        self._loop_profile_patches = ExitStack()

        def begin(module, inputs, kwargs):
            nonlocal current_chunk
            current_chunk += 1
            if current_chunk == chunk_index:
                self._loop_profile_chunk = {
                    "chunk_index": current_chunk,
                    **position_summary(module, inputs, kwargs),
                }
                self._loop_profiler.start()
                self._loop_profile_active = True

        def end(module, inputs, output):
            if self._loop_profile_active:
                self._loop_benchmark_finish_profile()

        self._loop_profile_handles.append(
            loop.register_forward_pre_hook(begin, with_kwargs=True)
        )

        def profile_function(function, name):
            @wraps(function)
            def wrapped(*args, **kwargs):
                if self._loop_profile_active:
                    with torch.profiler.record_function(f"loop_stage/{name}"):
                        return function(*args, **kwargs)
                return function(*args, **kwargs)

            return wrapped

        for name in (
            "normalize",
            "log_sinkhorn",
            "pre_aggregate",
            "residual_merge",
            "expand_streams",
            "collapse_streams",
        ):
            self._loop_profile_patches.enter_context(
                patch.object(
                    mhc_ops, name, profile_function(getattr(mhc_ops, name), name)
                )
            )

        def stage_hooks(name):
            contexts = []

            def enter(module, inputs):
                context = None
                if self._loop_profile_active:
                    context = torch.profiler.record_function(f"loop_stage/{name}")
                    context.__enter__()
                contexts.append(context)

            def leave(module, inputs, output):
                context = contexts.pop()
                if context is not None:
                    context.__exit__(None, None, None)

            return enter, leave

        for name, module in model.named_modules():
            stage = None
            if isinstance(module, SenseNovaLoopMHCRouter):
                stage = "mhc"
            elif name.endswith(".linear_attn"):
                stage = "gdn"
            elif name.endswith(".self_attn"):
                stage = "attention"
            elif name.endswith(".mlp"):
                stage = "mlp"
            if stage is not None:
                enter, leave = stage_hooks(stage)
                self._loop_profile_handles.extend(
                    [
                        module.register_forward_pre_hook(enter),
                        module.register_forward_hook(leave, always_call=True),
                    ]
                )
        self._loop_profile_handles.append(
            loop.register_forward_hook(end, always_call=True)
        )

    def _loop_benchmark_finish_profile(self) -> None:
        try:
            if self._loop_profile_active:
                self._loop_profiler.stop()
        finally:
            self._loop_profile_active = False
            for handle in self._loop_profile_handles:
                handle.remove()
            self._loop_profile_handles = []
            self._loop_profile_patches.close()

    def loop_benchmark_profile_stop(self) -> dict[str, Any]:
        self._loop_benchmark_finish_profile()
        if self._loop_profile_chunk is None:
            raise RuntimeError("Requested prefill chunk was not executed")
        events = self._loop_profiler.key_averages()
        rows = [
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "self_cuda_us": event.self_device_time_total,
                "cuda_us": event.device_time_total,
            }
            for event in sorted(
                events, key=lambda e: e.self_device_time_total, reverse=True
            )
            if event.self_device_time_total > 0 or event.key.startswith("loop_stage/")
        ]
        return {"chunk": self._loop_profile_chunk, "events": rows}
