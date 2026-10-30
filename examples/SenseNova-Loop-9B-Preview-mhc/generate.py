# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Text and image generation with the SenseNova Loop checkpoint."""

import argparse
import json
from pathlib import Path

import yaml

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override the checkpoint path in config.yaml")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path or URL; repeat for multiple images",
    )
    parser.add_argument(
        "--prompt-tokens", type=int, help="Repeat text to this token count"
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, choices=(4, 8))
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    args = parser.parse_args()
    if args.image and args.prompt_tokens is not None:
        parser.error(
            "--prompt-tokens cannot be used with --image; use --prompt instead"
        )
    config_path = Path(__file__).with_name("config.yaml")
    kwargs = {
        key.replace("-", "_"): value
        for key, value in yaml.safe_load(config_path.read_text()).items()
    }
    if args.model:
        kwargs["model"] = args.model
    for key in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
    ):
        if (value := getattr(args, key)) is not None:
            kwargs[key] = value
    if args.prompt_tokens is not None and not (
        0 < args.prompt_tokens < kwargs["max_model_len"]
    ):
        parser.error(
            f"--prompt-tokens must be between 1 and {kwargs['max_model_len'] - 1}"
        )
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    content = args.prompt
    images = []
    if args.image:
        from vllm.multimodal.utils import fetch_image

        content = []
        for source in args.image:
            if "://" not in source and not source.startswith("data:"):
                source = Path(source).expanduser().resolve().as_uri()
            images.append(fetch_image(source))
            content.append({"type": "image_url", "image_url": {"url": source}})
        content.append({"type": "text", "text": args.prompt})
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if args.prompt_tokens is not None:
        prompt = (prompt * ((args.prompt_tokens + len(prompt) - 1) // len(prompt)))[
            : args.prompt_tokens
        ]
    request = {"prompt_token_ids": prompt}
    if images:
        request["multi_modal_data"] = {"image": images}
    outputs = llm.generate(
        request,
        SamplingParams(
            temperature=0, max_tokens=args.max_tokens, ignore_eos=args.ignore_eos
        ),
        use_tqdm=False,
    )
    for output in outputs:
        result = output.outputs[0]
        print(
            json.dumps(
                {
                    "request_id": output.request_id,
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(result.token_ids),
                    "finished": output.finished,
                    "finish_reason": result.finish_reason,
                    "stop_reason": result.stop_reason,
                    "token_ids": result.token_ids,
                    "text": result.text,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
