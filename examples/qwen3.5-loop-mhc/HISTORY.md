# Qwen3.5 Loop + mHC：单机八卡文本推理

使用原始 `ckpt-225-bf16`，架构名保持 `Qwen3_5LoopForConditionalGeneration`。
环境为仓库的 `h800conda.sh`，需要八张可用 H800。配置采用 TP=8、PP=1、DP=1、
BF16 主干、FP32 mHC router 和 SSM 状态，以及 V2 runner 的 eager 文本模式。
输入与输出合计上限为 262144 tokens；分块 prefill 的单步预算为 8192 tokens。
完整历史仍按执行层保留，prefix caching 关闭。

Loop 模型默认启用 `VLLM_QWEN3_5_MHC_BACKEND=triton` 和
`VLLM_QWEN3_5_MHC_FUSION=all`，服务端与离线生成均无需额外设置。
需要完整回退原 PyTorch 路径时，同时设置 `VLLM_QWEN3_5_MHC_BACKEND=torch`
和 `VLLM_QWEN3_5_MHC_FUSION=none`。

## 128K 单 token TTFT 与 Sinkhorn 融合

`VLLM_QWEN3_5_MHC_BACKEND` 默认 `triton`，使用 Qwen Loop 专用的 FP32 log-Sinkhorn
kernel；设为 `torch` 可回退 Sinkhorn。该融合只覆盖 CUDA 上无梯度、FP32、四流、20 轮的
三维路由矩阵，其他输入回退并记录日志。它保留先行后列的归一化顺序、FP32 投影、
BF16 中间转换和残差运算，不复用数学定义不同的通用 mHC kernel。
归约顺序按本仓库 PyTorch 2.13 CUDA 实现对齐；升级 PyTorch、Triton 或更换设备后，
应重新运行数值验收再启用。有限用例零差异不代表所有输入的逐位一致性证明。

从 SenseSpec 根目录运行 kernel 正确性测试：

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python -m pytest \
  --confcutdir=vllm/tests/kernels vllm/tests/kernels/test_qwen3_5_mhc.py -q
```

在空闲八卡上启动本机基准服务（诊断扩展和开发接口只用于本次本机验收）：

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_QWEN3_5_MHC_BACKEND=torch \
  VLLM_QWEN3_5_MHC_FUSION=none bash vllm/examples/qwen3.5-loop-mhc/run.sh \
  --host 127.0.0.1 --port 18328 --max-model-len 131072 \
  --worker-extension-cls benchmarks.qwen3_5_loop_benchmark_worker.LoopBenchmarkWorker
```

另一个终端按顺序运行两个版本，`--model` 与服务使用同一 checkpoint 路径：

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/benchmarks/benchmark_qwen3_5_loop.py \
  --url http://127.0.0.1:18328 --model /path/to/checkpoint \
  --backend torch --profile --output reference.json
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/benchmarks/benchmark_qwen3_5_loop.py \
  --url http://127.0.0.1:18328 --model /path/to/checkpoint \
  --backend triton --profile --output triton.json
```

工具核对 TP=8、131072 上限、8192 分块、并发一、eager、prefix caching 关闭、
BF16 主干、FP32 SSM 和无推测解码，比较客户端源码与 worker 初始化时的源码身份。
主请求为固定的 131071 输入加 1 输出，贪心、忽略 EOS、返回 token IDs 与 logprobs。
TTFT 从客户端 POST 调用开始，到收到包含生成 token ID 的 SSE 事件为止；
控制事件和空 usage 事件不会结束计时，空文本的特殊 token 仍计为输出。

每版单列预热，随后至少五次无 profiler 测量，保存逐次值、中位数、范围、
引擎指标增量、实际后端调用计数、各 rank CUDA 显存峰值，并在长请求之后运行短请求。
引擎指标只有计数增量恰为一次时才归属于该请求，否则记为 `null`。
显存包含已分配的缓存池，不代表单请求的 KV 大小。`--profile` 在测速后另发请求，
只分析第一个 8192-token chunk 的 GPU 算子及 mHC、GDN、attention、MLP 阶段；
该诊断时间不参与 TTFT 汇总，也不能代表全部 128K 的 GPU 分解。

`triton` 测速前先在同一引擎运行两次参考路径、一次候选路径，记录每个 chunk 的
五个边界特征、独立 post-norm 最终状态和目标 logits 的 dtype、形状、有限值及
原始字节 SHA256。非输出 TP rank 可以没有 logits；至少一个 rank 必须提供 logits。
参考不可重复、存在非有限值、候选哈希不同或实际回退都会阻止融合性能验收并恢复
所选参考路径（默认 `torch`）。首次候选诊断的请求耗时单列，其中可能包含首次 JIT 及采集开销，不能作为
纯编译耗时或正式 TTFT。常规服务启用融合不运行上述昂贵的完整模型审计，应先完成验收。

普通生成默认启用两阶段融合，无需诊断扩展。仅复现第一阶段时设置
`VLLM_QWEN3_5_MHC_FUSION=none`，再用 `VLLM_QWEN3_5_MHC_BACKEND` 对照 Sinkhorn。
测试服务结束后应正常停止，释放八卡资源。

2026-10-12 八卡 H800 验收：五次稳态 TTFT 中位数从 19.688 秒降至 16.937 秒，
降低 13.97%（1.162×）；完整 128K 请求的特征、最终状态和 logits 字节哈希对照通过。
原始数据：[参考路径](performance/reference.json.gz)、[融合路径](performance/triton.json.gz)。
文件使用 gzip 压缩，解压后是完整 JSON，包含数值审计、逐次计时、配置、源码身份及 profiling。

## 继续融合与严格 128K 输入

2026-10-15 在 8 × H800、TP=8 上完成 131072 输入 + 1 输出的实测：
相对已启用 Triton Sinkhorn 的基线，五次 TTFT 中位数从 16.958560 秒降至
14.029082 秒，减少 17.27%；完整请求的已覆盖数值审计逐字节一致。
详见 [第二阶段结果](../../../docs/qwen3_5_loop_mhc_fusion_results.md)及
[原始数据](performance/fusion-20261015/)。

`VLLM_QWEN3_5_MHC_FUSION` 独立选择非 Sinkhorn 的局部融合，默认 `all`。
可选 `none`（关闭）、`norm`（归一化搬运）、`pre`（四流聚合）、`merge`（残差回写）或 `all`。
归一化保留原 FP32 mean 和 rsqrt；残差回写保留原矩阵乘和 BF16 乘法的中间舍入，
再融合缩放与相加。CPU、不支持的布局或梯度输入使用原路径并记录回退。
先在目标设备、软件版本和权重上完成下述审计，再为常规服务选择融合。

本轮的 128K 指 **131072 输入加 1 输出**，因此服务总长度为 131073。
从 SenseSpec 根目录、在空闲八卡上启动：

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_QWEN3_5_MHC_BACKEND=triton VLLM_QWEN3_5_MHC_FUSION=none \
  bash vllm/examples/qwen3.5-loop-mhc/run.sh \
  --host 127.0.0.1 --port 18428 --max-model-len 131073 \
  --gpu-memory-utilization 0.92 \
  --worker-extension-cls benchmarks.qwen3_5_loop_benchmark_worker.LoopBenchmarkWorker
```

另一个终端中，以已经启用 Triton Sinkhorn 的当前实现作为参考：

```bash
source h800conda.sh
loop_checkpoint=/path/to/checkpoint
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/benchmarks/benchmark_qwen3_5_loop.py \
  --url http://127.0.0.1:18428 --model "$loop_checkpoint" \
  --prompt-tokens 131072 --max-model-len 131073 \
  --backend triton --reference-backend triton --fusion none \
  --profile --profile-chunks 0 7 15 --output reference.json
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/benchmarks/benchmark_qwen3_5_loop.py \
  --url http://127.0.0.1:18428 --model "$loop_checkpoint" \
  --prompt-tokens 131072 --max-model-len 131073 \
  --backend triton --reference-backend triton --fusion all \
  --profile --profile-chunks 0 7 15 --output fused.json
```

两组都核验 16 个分块的实际位置和完整特征行数。候选检查五份边界特征、最终状态、
logits 与两次参考的原始字节哈希，并确认每个 rank 的各项融合确实执行、没有回退。
每个选定分块各用独立请求诊断；`profiles` 字段含位置元数据和算子统计，覆盖归一化、
聚合、残差回写、流展开/收敛以及主干阶段。嵌套范围不能直接相加，三个采样块也不能
代表完整请求的 GPU 时间占比。正式 TTFT 请求没有诊断或采集 hooks。

算子微基准位于 `vllm/benchmarks/kernels/benchmark_qwen3_5_mhc.py`，会先检查输出字节
一致性，再分别测量参考与融合算子；它的加速比不能代替完整请求 TTFT。
常规服务默认使用 `VLLM_QWEN3_5_MHC_BACKEND=triton VLLM_QWEN3_5_MHC_FUSION=all`，
不需要诊断扩展；将 `VLLM_QWEN3_5_MHC_FUSION=none` 即恢复仅有 Sinkhorn 融合的路径。

## 常规生成与历史验收

从 SenseSpec 根目录生成短文本：

```bash
source h800conda.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/examples/qwen3.5-loop-mhc/generate.py --prompt '请用一句话介绍你自己。'
```

使用实际 token 数检查 256K 总长度，保留 16 个生成 token：

```bash
uv run --no-project --python .venv/bin/python .venv/bin/python \
  vllm/examples/qwen3.5-loop-mhc/generate.py \
  --prompt-tokens 262128 --max-tokens 16 --ignore-eos
```

持续解码用例使用 `--prompt-tokens 258048 --max-tokens 4096 --ignore-eos`，
输入与输出同样合计 262144 tokens。8K 回归可使用
`--prompt-tokens 8176 --max-tokens 16 --ignore-eos`。

脚本使用 checkpoint 的 tokenizer 和 chat template，通过标准 `LLM.generate`
输出输入/输出 token 数、token IDs、完成状态和文本。`--ignore-eos` 仅用于长度压力运行；
普通文本运行沿用 `generation_config.json`，其中 EOS 包含 `248044` 和 `248046`。
可用 `--model /path/to/checkpoint` 覆盖默认路径，原权重无需转换或修改。

通过标准 vLLM OpenAI API 服务入口部署：

```bash
bash vllm/examples/qwen3.5-loop-mhc/run.sh --host 127.0.0.1 --port 8000
```

`config.yaml` 同时供服务和离线脚本使用。默认 `max-num-seqs=1`；少量变长请求可在同一
引擎中排队运行。分块 prefill 允许批次 token 预算小于上下文上限；
非分块 prefill 仍要求 `max-num-batched-tokens >= max-model-len`。

模型按 `[0..3] + [4..27] * 3 + [28..31]` 执行，共 80 个执行位置；
同一 TP rank 上的 32 组物理权重共享，20 个 attention 和 60 个 GDN 执行状态各自独立。
每轮 mHC 都展开四流并在末尾取均值收敛，路由器在各 TP rank 完整复制。
加载后各 rank 会记录权重共享、路由器精度及执行状态对象检查结果。

当前配置校验会拒绝其他并行拓扑、量化、LoRA、推测解码、多模态输入、prefix caching、
非 eager 运行和超过 262144 的模型长度。这里的功能与结构验证不代表
与训练实现或其他推理框架的数值等价性证明。

2026-10-10 已在八张 H800 上通过 256K 功能与结构验收，主要结果如下：

| 256K 用例 | 八个 rank 的实际 prefill | prefill 耗时 | 请求总耗时 |
| --- | --- | --- | --- |
| 262128 输入 + 16 输出 | 32 块，最后一块 8176 tokens | 40.17 秒 | 45.64 秒 |
| 258048 输入 + 4096 输出 | 32 块，最后一块 4096 tokens | 39.70 秒 | 1532.46 秒（约 25.54 分钟） |

两组长请求之后均在同一引擎完成 15 输入 + 16 输出的短请求。8K 非分块回归、
8191/8192/8193/16386 输入的分块边界、32K/64K/128K 总长度、剩余输出空间及
两个 EOS 停止用例也通过，所有生成用例所选 token 的 logprob 均为有限值。
各 rank 保持 20 个 attention 和 60 个 GDN 状态，attention/卷积状态为 BF16、SSM 为 FP32。
记录的 CUDA 峰值 allocated 约为每卡 72.68 GiB，包含启动 profiling 和预分配缓存池，
不能作为单条 256K 请求所需的 KV 大小。长输入由重复 token 构造，验收未评估长文本质量或数值对齐。

配置回归 5 项及相关 pre-commit / Ruff 检查通过。末尾 262145-token 负例已被引擎正确拒绝，
但诊断脚本首次只捕获 `ValueError`，未捕获当前 vLLM 的 `VLLMValidationError`，因此原始进程退出码为 1。
报告保留原始失败状态和 traceback，并依据 17 个已通过请求及同一输入校验路径的 CPU 复核汇总为通过；
未重跑已通过的长请求。运行结束后 GPU 已释放。

以下为 8K 非分块配置的历史验收。2026-10-10 在八张 H800 上通过原 checkpoint 的 `LLM.generate` / `LLM.chat`
完成功能验收，主要结果如下：

| 场景 | 实际结果 |
| --- | --- |
| 权重与执行状态 | 八个 rank 均通过共享参数、FP32 router 校验；每 rank 注册 20 个 attention 和 60 个 GDN 状态，SSM 为 FP32 |
| 短请求与 8K | 15 输入 + 16 输出；8176 输入 + 16 输出，均完成多步生成 |
| 总长度约束 | 8188 输入、申请 16 输出，实际生成 4；非法 token 和 8193 长输入被拒绝 |
| 停止行为 | 分别强制采样两个 EOS，均以 `stop` 完成；显式停止 token 也生效 |
| 变长与复用 | 32/257/2048 输入分别返回 4/6/8 输出，随后 15 输入 + 8 输出成功 |
| 正常聊天 | 重启后使用默认 EOS 设置完成 12 输入 + 16 输出，以长度预算结束 |
| 配置边界 | TP=4、FP16、LoRA、推测解码、多模态、非 eager、两个缓存开关、超长上下文及 BF16 SSM 配置均被拒绝 |
| 标准 Qwen3.5 回归 | 同一示例以标准 `Qwen3_5ForConditionalGeneration` checkpoint 完成 15 输入 + 16 输出，进程正常退出 |

上述生成请求所选 token 的 logprob 均为有限值。EOS 场景验证停止机制，不代表模型自然
生成 EOS 的概率；短样例也不用于评价模型质量。现有 Qwen3.5 注册用例为 8 项通过，
vLLM 暂存文件 hooks 与本次新增代码的类型检查通过。

缓存诊断首次直接返回 `torch.dtype` 时触发了既有 RPC 序列化限制。重启后在 worker 内
转换为可序列化摘要完成诊断；诊断进程的额外序列化开关未写入启动配置。八个 GPU worker
正常退出，EngineCore 父进程由既有超时机制清理，之后每卡仅占用 4 MiB。

完整 TorchSpec 测试命令 `pytest tests --maxfail=1 --timeout=120 -q` 运行至 69 项通过、
1 项失败、30 个 subtests 通过后停止，后续用例未执行。失败为既有全局 Ray mock
影响 `test_capacity_handling.py::TestAsyncTrainingControllerPoolBytes::test_initial_pool_bytes_is_zero`，
抛出 `module 'ray' has no attribute '_private'`。另一个既有问题是 vLLM 全局注册覆盖检查
缺少三个 `Glm5Next` 架构的测试条目。本次未修改这些无关测试，因此不宣称全套测试通过。
