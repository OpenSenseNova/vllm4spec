#!/usr/bin/env bash
set -euo pipefail

loop_example_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
loop_repo_dir=$(cd -- "${loop_example_dir}/../../.." && pwd)
loop_python="${SENSESPEC_ENV:-${loop_repo_dir}/.venv}/bin/python"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V2_MODEL_RUNNER=1
exec "${loop_python}" -m sensespec.runtime -- \
    "${loop_python}" -m vllm.entrypoints.openai.api_server \
    --config "${loop_example_dir}/config.yaml" "$@"
