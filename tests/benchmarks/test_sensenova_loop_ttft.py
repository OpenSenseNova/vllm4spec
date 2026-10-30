# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any

import httpx
import pytest

from benchmarks.benchmark_sensenova_loop import audit_outputs, measure_request


def test_ttft_ignores_control_events_and_counts_empty_text_token(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(
        "benchmarks.benchmark_sensenova_loop.time.perf_counter", lambda: now[0]
    )

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for timestamp, event in [
                (1.0, {"choices": [{"text": "", "token_ids": []}]}),
                (
                    3.0,
                    {
                        "choices": [
                            {
                                "text": "",
                                "token_ids": [248046],
                                "finish_reason": "length",
                            }
                        ]
                    },
                ),
                (
                    5.0,
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                    },
                ),
            ]:
                now[0] = timestamp
                yield f"data: {json.dumps(event)}\n\n".encode()
            now[0] = 6.0
            yield b"data: [DONE]\n\n"

    def handle(request):
        body = json.loads(request.content)
        assert body["prompt"] == [1, 2, 3]
        assert body["max_tokens"] == 1 and body["ignore_eos"]
        return httpx.Response(200, stream=Stream())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = measure_request(client, "http://localhost", "loop", [1, 2, 3])
    assert result["ttft_seconds"] == 3.0
    assert result["request_seconds"] == 6.0
    assert result["token_ids"] == [248046]


@pytest.mark.parametrize(
    "failure", ["usage_only", "truncated", "wrong_length", "server_error"]
)
def test_incomplete_stream_is_rejected_instead_of_reporting_ttft(failure):
    events: list[dict[str, Any]] = [
        {"choices": [{"token_ids": [42], "finish_reason": "length"}]}
    ]
    usage: dict[str, Any] = {
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }
    if failure == "usage_only":
        events = []
    if failure == "wrong_length":
        usage["usage"]["prompt_tokens"] = 2
    if failure == "server_error":
        events = [{"error": {"message": "failed"}}]
    events.append(usage)
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if failure != "truncated":
        body += "data: [DONE]\n\n"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    with httpx.Client(transport=transport) as client, pytest.raises(RuntimeError):
        measure_request(client, "http://localhost", "loop", [1, 2, 3])


@pytest.mark.parametrize(
    "failure",
    [
        "nonfinite_logits",
        "fallback",
        "fusion_fallback",
        "missing_states",
        "missing_rows",
        "wrong_positions",
        "missing_rank",
        None,
    ],
)
@pytest.mark.parametrize("tp_size", [4, 8])
def test_numerical_audit_rejects_invalid_or_unfused_outputs(failure, tp_size):
    backend = ["torch"]
    fusion = ["none"]

    def handle(request):
        body = json.loads(request.content)
        if request.url.path == "/v1/completions":
            events = [
                {"choices": [{"token_ids": [42], "finish_reason": "length"}]},
                {"usage": {"prompt_tokens": 3, "completion_tokens": 1}},
            ]
            text = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(200, text=text + "data: [DONE]\n\n")
        method = body["method"]
        results: list[Any] = [None] * tp_size
        if method == "loop_benchmark_backend":
            backend[0] = body["args"][0]
            fusion[0] = body["args"][1]
        elif method == "loop_benchmark_status":
            actual = "torch" if failure == "fallback" else backend[0]
            results = [
                {
                    "sinkhorn_calls": {
                        "torch": int(actual == "torch"),
                        "triton": int(actual == "triton"),
                    },
                    "fusion_modes": [fusion[0]],
                    "fusion_calls": {
                        name: {
                            "torch": int(
                                fusion[0] == "none" or failure == "fusion_fallback"
                            ),
                            "triton": int(
                                fusion[0] == "all" and failure != "fusion_fallback"
                            ),
                        }
                        for name in ("normalize", "pre_aggregate", "residual_merge")
                    },
                }
            ] * tp_size
        elif method == "loop_benchmark_capture_stop":
            states = {
                name: [{"finite": True, "sha256": "equal", "shape": [3, 4096]}]
                for name in (
                    "layer_3",
                    "layer_27",
                    "layer_51",
                    "layer_53",
                    "layer_54",
                    "final_post_norm",
                    "logits",
                )
            }
            states["positions"] = [
                {"tokens": 3, "start": 0, "end": 2, "contiguous": True}
            ]
            if failure == "nonfinite_logits":
                states["logits"][0]["finite"] = False
            if failure == "missing_states":
                del states["layer_27"]
            if failure == "missing_rows":
                states["layer_27"][0]["shape"] = [2, 4096]
            if failure == "wrong_positions":
                states["positions"][0]["start"] = 1
            results = [states] * (tp_size - int(failure == "missing_rank"))
        return httpx.Response(200, json={"results": results})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = audit_outputs(
            client,
            "http://localhost",
            "loop",
            [1, 2, 3],
            capture_layers=[3, 27, 51, 53, 54],
            tp_size=tp_size,
            candidate_fusion="all",
        )
    assert result["candidate_equal"] == (failure is None)
    assert backend[0] == "torch"


def test_capture_uses_the_56_layer_model_boundaries(monkeypatch):
    import torch

    from benchmarks import sensenova_loop_benchmark_worker as worker_module

    class Loop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(56))
            self.norm = torch.nn.Identity()
            self.capture_boundary_slots = {1, 2, 3, 4, 28, 52, 53, 54, 55}
            for index, layer in enumerate(self.layers):
                layer.expand_streams = index in (4, 28)
                layer.collapse_streams = index in (27, 51)

        def forward(self, input_ids, positions):
            hidden = torch.ones(len(positions), 8)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.norm(hidden)

    model = torch.nn.Module()
    model.loop = Loop()
    model.logits_processor = torch.nn.Identity()
    worker = worker_module.LoopBenchmarkWorker()
    monkeypatch.setattr(worker, "get_model", lambda: model, raising=False)
    monkeypatch.setattr(worker_module, "SenseNovaLoopModel", Loop)

    worker.loop_benchmark_capture_start()
    model.logits_processor(model.loop(None, positions=torch.arange(2)))
    captured = worker.loop_benchmark_capture_stop()
    assert set(captured) == {
        "layer_3",
        "layer_27",
        "layer_51",
        "layer_53",
        "layer_54",
        "final_post_norm",
        "logits",
        "positions",
    }
    for name, values in captured.items():
        if name != "positions":
            assert len(values) == 1 and values[0]["shape"] == [2, 8]
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in model.modules()
    )


@pytest.mark.parametrize("fail", [False, True])
def test_selected_chunk_profile_restores_functions_and_hooks(monkeypatch, fail):
    import torch

    from benchmarks import sensenova_loop_benchmark_worker as worker_module
    from vllm.model_executor.layers import sensenova_loop_mhc as mhc_ops

    class Loop(torch.nn.Module):
        def forward(self, input_ids, positions):
            result = mhc_ops.normalize(torch.ones(len(positions), 4, 8))
            if fail and int(positions[0]) == 4:
                raise RuntimeError("model failure")
            return result

    class Profiler:
        starts = 0
        stops = 0

        def start(self):
            self.starts += 1

        def stop(self):
            self.stops += 1

        def key_averages(self):
            return []

    model = Loop()
    profiler = Profiler()
    worker = worker_module.LoopBenchmarkWorker()
    monkeypatch.setattr(worker, "get_model", lambda: model, raising=False)
    monkeypatch.setattr(worker_module, "SenseNovaLoopModel", Loop)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: profiler)
    original = mhc_ops.normalize

    worker.loop_benchmark_profile_start(1)
    model(None, positions=torch.arange(4))
    assert profiler.starts == profiler.stops == 0
    if fail:
        with pytest.raises(RuntimeError, match="model failure"):
            model(None, positions=torch.arange(4, 8))
    else:
        model(None, positions=torch.arange(4, 8))
    result = worker.loop_benchmark_profile_stop()

    assert result["chunk"] == {
        "chunk_index": 1,
        "tokens": 4,
        "start": 4,
        "end": 7,
        "contiguous": True,
    }
    assert profiler.starts == profiler.stops == 1
    assert mhc_ops.normalize is original
    assert not model._forward_pre_hooks and not model._forward_hooks
    model(None, positions=torch.arange(8, 12))
    assert profiler.starts == profiler.stops == 1


@pytest.mark.parametrize("multimodal", [False, True])
def test_position_coverage_accepts_text_mrope_but_rejects_different_axes(multimodal):
    import torch

    from benchmarks.sensenova_loop_benchmark_worker import position_summary

    class Model:
        def forward(self, input_ids, positions):
            pass

    positions = torch.arange(8192, 16384).repeat(3, 1)
    if multimodal:
        positions[1, -1] -= 1
    observed = position_summary(
        Model(), (), {"input_ids": None, "positions": positions}
    )
    assert observed == {
        "tokens": 8192,
        "start": 8192,
        "end": 16383,
        "contiguous": not multimodal,
    }
