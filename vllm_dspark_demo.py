# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import sys
from pathlib import Path

from vllm_demo import SOURCE_MODEL_PATH, resolve_model_path

DEEPSPEC_TARGET_LAYER_IDS = (2, 22, 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GLM-5.3-Flash hidden states for DSpark training."
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/vllm-dspark-features"),
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句话介绍投机解码。",
    )
    parser.add_argument("--train-data", nargs="+", type=Path)
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-logits", action="store_true")
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument(
        "--target-layer-ids",
        nargs="+",
        type=int,
        default=list(DEEPSPEC_TARGET_LAYER_IDS),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    args = parser.parse_args()
    for name in (
        "max_length",
        "max_num_batched_tokens",
        "tensor_parallel_size",
        "min_loss_tokens",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.target_layer_ids != sorted(set(args.target_layer_ids)) or not all(
        0 <= layer < 44 for layer in args.target_layer_ids
    ):
        parser.error("--target-layer-ids must be unique, sorted decoder IDs in [0, 43]")
    return args


def load_final_norm(model_path):
    from safetensors import safe_open

    config = json.loads((model_path / "config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    name = "model.language_model.norm.weight"
    with safe_open(model_path / index["weight_map"][name], framework="pt") as handle:
        weight = handle.get_tensor(name)
    return weight, config["text_config"]["rms_norm_eps"]


def convert_hidden_states(tensors, input_ids, norm_weight, norm_eps, num_layers):
    """Convert ordered auxiliary outputs to the DSpark teacher tensor contract."""
    import torch

    hidden = tensors["hidden_states"]
    expected = (input_ids.numel(), num_layers + 1, norm_weight.numel())
    if not torch.equal(tensors["token_ids"], input_ids):
        raise ValueError("Extracted token IDs do not match the training input IDs")
    if tuple(hidden.shape) != expected:
        raise ValueError(f"Expected hidden-state shape {expected}, got {hidden.shape}")
    last_hidden = torch.empty_like(hidden[:, -1])
    for start in range(0, hidden.shape[0], 2048):
        chunk = hidden[start : start + 2048]
        if not torch.isfinite(chunk).all():
            raise ValueError("Extracted hidden states contain non-finite values")
        last = chunk[:, -1].float()
        last = last * torch.rsqrt(last.square().mean(-1, keepdim=True) + norm_eps)
        # Match CUDA RMSNorm's single BF16 cast after the weight multiplication.
        last_hidden[start : start + 2048] = (last * norm_weight.float()).to(
            hidden.dtype
        )
    return {
        "input_ids": input_ids,
        "target_hidden_states": hidden[:, :-1].flatten(1).contiguous(),
        "target_last_hidden_states": last_hidden,
    }


def write_training_cache(args, llm, extract):
    """Reuse DeepSpec's preprocessing and v4 cache format (text-only, CP=1)."""
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from deepspec.data.target_cache_dataset import (
        CacheDataset,
        ConversationCollator,
        LocalTargetCacheWriter,
        build_source_jsonl_fingerprints,
        build_target_cache_manifest,
        write_target_cache_manifest,
    )

    sources = sorted(str(path.resolve()) for path in args.train_data)
    fingerprints = build_source_jsonl_fingerprints(sources)
    collator = ConversationCollator(
        tokenizer=llm.get_tokenizer(),
        chat_template="glm5_next",
        max_length=args.max_length,
        min_loss_tokens=args.min_loss_tokens,
    )
    writer = LocalTargetCacheWriter(
        rank_dir=str(args.output_dir),
        max_shard_bytes=4 * 1024**3,
    )
    try:
        for source in sources:
            with open(source) as handle:
                for line in handle:
                    if (
                        args.max_samples is not None
                        and writer.num_local_samples >= args.max_samples
                    ):
                        break
                    batch = collator([json.loads(line)])
                    if batch is None:
                        continue
                    sample = {key: value[0] for key, value in batch.items()}
                    features = extract(sample["input_ids"])
                    writer.write_sample(
                        sample_id=writer.num_local_samples,
                        attention_mask=sample["attention_mask"],
                        loss_mask=sample["loss_mask"],
                        **features,
                    )
                    print(
                        f"[dspark-cache] samples={writer.num_local_samples} "
                        f"tokens={sample['input_ids'].numel()} "
                        f"loss_tokens={int(sample['loss_mask'].sum())}",
                        flush=True,
                    )
    finally:
        writer.close()
    if not writer.num_local_samples:
        raise ValueError(
            "No samples passed preprocessing and minimum loss-token filtering"
        )
    manifest = build_target_cache_manifest(
        num_samples=writer.num_local_samples,
        shards=[
            {"shard_id": i, "file_name": name}
            for i, name in enumerate(writer.local_shard_files)
        ],
        target_layer_ids=args.target_layer_ids,
        hidden_size=4096,
        extra_fields={
            "target_model_name_or_path": str(
                args.model.expanduser().resolve() if args.model else SOURCE_MODEL_PATH
            ),
            "source_jsonl_paths": sources,
            "source_jsonl_fingerprints": fingerprints,
            "chat_template": "glm5_next",
            "max_length": args.max_length,
            "min_loss_tokens": args.min_loss_tokens,
            "cache_context_parallel_size": 1,
            "context_layout": "contiguous",
            "index_files": ["samples.local.idx"],
            "stores_target_last_hidden_states": True,
            "target_execution": "full_model",
            "target_execution_num_hidden_layers": 45,
            "target_final_hidden_layer": 44,
            "target_final_hidden_source": "full_model_final_norm",
            "target_backend": "vllm",
            "target_quantization": "fp8",
            "target_activation_dtype": "bfloat16",
            "target_tensor_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "source_sample_limit": args.max_samples,
        },
    )
    write_target_cache_manifest(output_dir=str(args.output_dir), manifest=manifest)
    dataset = CacheDataset(str(args.output_dir))
    print(f"[dspark-cache] CacheDataset validated: samples={len(dataset)}", flush=True)
    dataset.close()


def main() -> None:
    import torch

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector,
    )

    args = parse_args()
    if args.train_data and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Use a new, empty cache output directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(args.model)
    norm_weight, norm_eps = load_final_norm(model_path)
    aux_layers = [layer + 1 for layer in args.target_layer_ids] + [45]

    print(
        f"[dspark-demo] model={model_path} tp={args.tensor_parallel_size} "
        f"layers={aux_layers} max_length={args.max_length}",
        flush=True,
    )
    llm = LLM(
        model=str(model_path),
        load_format="instanttensor",
        tensor_parallel_size=args.tensor_parallel_size,
        # The connector needs one sampled token to finish the prefill request.
        max_model_len=args.max_length + 1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=0.7,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": aux_layers,
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": str(args.output_dir / "raw"),
            },
        ),
    )

    lm_head = None
    if args.verify_logits:
        from safetensors import safe_open

        index = json.loads((model_path / "model.safetensors.index.json").read_text())
        with safe_open(
            model_path / index["weight_map"]["lm_head.weight"], framework="pt"
        ) as handle:
            lm_head = handle.get_tensor("lm_head.weight").to("cuda")

    def extract(input_ids):
        output = llm.generate(
            [{"prompt_token_ids": input_ids.tolist()}],
            SamplingParams(
                temperature=0.0,
                max_tokens=1,
                logprobs=20 if args.verify_logits else None,
            ),
            use_tqdm=False,
        )[0]
        path = output.kv_transfer_params["hidden_states_path"]
        tensors = example_hidden_states_connector.load_hidden_states(path)
        features = convert_hidden_states(
            tensors,
            input_ids,
            norm_weight,
            norm_eps,
            len(args.target_layer_ids),
        )
        if lm_head is not None:
            with torch.inference_mode():
                last = features["target_last_hidden_states"][-1:].to("cuda")
                logits = torch.nn.functional.linear(last, lm_head).float()[0]
                logprobs = logits.log_softmax(-1)
                expected = output.outputs[0].logprobs[0]
                error = max(
                    abs(logprobs[token].item() - value.logprob)
                    for token, value in expected.items()
                )
                print(
                    f"[dspark-check] tokens={input_ids.numel()} "
                    f"top20_logprob_max_abs={error:.6f}",
                    flush=True,
                )
                if error > 0.1:
                    raise ValueError(
                        "Exported final features do not reproduce teacher logits"
                    )
        example_hidden_states_connector.cleanup_hidden_states(
            path,
            keep_hidden_states=not bool(args.train_data),
        )
        if not args.train_data:
            print(f"raw_hidden_states={path}", flush=True)
        return features

    if args.train_data:
        write_training_cache(args, llm, extract)
    else:
        from safetensors.torch import save_file

        input_ids = torch.tensor(llm.get_tokenizer().encode(args.prompt))
        features = extract(input_ids)
        path = args.output_dir / "dspark.safetensors"
        save_file(features, path)
        print({key: tuple(value.shape) for key, value in features.items()})
        print(f"saved={path}")


if __name__ == "__main__":
    main()
