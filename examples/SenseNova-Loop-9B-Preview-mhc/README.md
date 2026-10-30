# SenseNova Loop + mHC 图文推理

目标权重为 `SenseNova-6.8-Loop-9B-Preview-mtp13200`，架构名为
`SenseNovaLoopForConditionalGeneration`。32 个物理层中的 `[4, 28)` 执行两轮，
形成 56 个执行层；mHC 使用两条残差流。默认 TP=4、PP=1、BF16 主干、
FP32 router 和 SSM 状态，启用 V2 runner、eager 和分块 prefill，关闭 prefix caching。
配置的上下文上限为 131072 tokens，这不是完整 128K 验收声明。
默认加载视觉编码器，支持纯文本及图文混合输入。每条请求最多 4 张图，每张
图片处理后的最大像素数为 1048576；视频在本示例中关闭。

完成 SenseSpec 根 README 的环境安装后，从 SenseSpec 根目录运行。脚本默认使用
`.venv` 和 GPU 0–3；`SENSESPEC_ENV` 可指定其他已安装环境，
`CUDA_VISIBLE_DEVICES` 可指定可用 GPU。两流 mHC 使用现有 PyTorch 回退；
四流 Triton 融合的历史加速结果不适用于该权重。

## 启动服务

```bash
bash vllm4spec/examples/SenseNova-Loop-9B-Preview-mhc/run.sh --host 127.0.0.1 --port 8000

# 小上下文验证；也可通过 --model 覆盖权重路径。
bash vllm4spec/examples/SenseNova-Loop-9B-Preview-mhc/run.sh \
  --host 127.0.0.1 --port 8000 --max-model-len 2048 \
  --max-num-batched-tokens 512 --gpu-memory-utilization 0.5
```

## 离线生成

通过 `sensespec.runtime` 配置本地 vLLM、动态库路径和 V2 runner：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m sensespec.runtime -- \
  .venv/bin/python vllm4spec/examples/SenseNova-Loop-9B-Preview-mhc/generate.py \
  --prompt '请用一句话介绍你自己。' --max-tokens 16 \
  --max-model-len 2048 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.5
```

使用 `--prompt-tokens` 可以构造固定长度输入。配置的长度和实际已验证长度应分开记录。

图片使用 `--image` 传入本地路径、HTTP(S) 或 data URL，多图可重复传入：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m sensespec.runtime -- \
  .venv/bin/python vllm4spec/examples/SenseNova-Loop-9B-Preview-mhc/generate.py \
  --image vllm4spec/tests/multimodal/assets/image1.png \
  --prompt '读出图中的文字。' --max-tokens 128 \
  --max-model-len 4096 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.5
```

`--prompt-tokens` 仅适用于纯文本。图文上下文长度包含展开后的图片 token。
在线服务使用 `/v1/chat/completions` 的标准内容列表，例如：

```json
{"model":"/mnt/afs_agents/yankuihao/model_ckpts/SenseNova-6.8-Loop-9B-Preview-mtp13200","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,<图片的base64内容>"}},{"type":"text","text":"读出图中的文字。"}]}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}
```

训练也已开启图文，见 [训练数据说明](../../../examples/data/README.md)。

## 运行选项

自定义环境变量现为 `VLLM_SENSENOVA_LOOP_MHC_BACKEND`（`torch`/`triton`）和
`VLLM_SENSENOVA_LOOP_MHC_FUSION`（`none`/`norm`/`pre`/`merge`/`all`）。
旧环境变量、旧 Python 模块路径和旧 Loop architecture 名均已退役。

## 本次迁移验证

2026-10-30：179 项相关测试通过；在四张 H800 上以 TP=4、2048 上下文上限
加载原始权重并完成文本生成。513-token 输入跨 512 + 1 两个 prefill 分块，
三次 PyTorch 请求的五份层特征、最终状态、logits 和输出 token 一致；基准脚本
完成五次独立测量。模型主体及 mHC 算子的语法树在名称替换后与原实现一致。
本次没有重跑 128K/256K 边界或完整训练。
