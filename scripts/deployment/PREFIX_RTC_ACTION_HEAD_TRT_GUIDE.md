# GR00T N1.7 Prefix-RTC Action-Head TensorRT 指南

本文记录 Task3 GR00T N1.7 checkpoint 在本机 RTX 4090 上的 TensorRT 接入、验证和回退方法。最终方案以本机现有代码和原始 PyTorch 链路为准，参考 RTX 5880 Ada 包的 shape 适配与对拍思路，但不直接覆盖参考包代码。

验证日期：2026-08-04。

## 1. 最终方案

默认仍为完整 PyTorch。只有显式设置以下变量才启用 TensorRT：

```text
GR00T_INFERENCE_BACKEND=tensorrt
GR00T_TRT_MODE=prefix_rtc_action_head
```

实际执行结构：

```text
图像与文本
  │
  ├─ ViT                         PyTorch
  ├─ LLM                         PyTorch
  ├─ VLLN / VL Self-Attention    PyTorch
  │
  └─ Prefix-RTC Action Head
       ├─ State Encoder          TensorRT
       ├─ Action Encoder         TensorRT
       ├─ DiT                    TensorRT
       └─ Action Decoder         TensorRT
```

没有使用 `llm_bf16.engine` 或 `vit.engine`。此前该 checkpoint 的 TRT LLM 在 Prefix-RTC 续接后缀中出现过明显发散，因此 `prefix_rtc_full_pipeline` 不作为生产选项。

## 2. 不改变现有链路的原则

- launcher 默认 `GR00T_INFERENCE_BACKEND=pytorch`。
- PyTorch 模式不会加载任何 TensorRT engine。
- 只通过环境变量选择 TensorRT，不改变 GR00T、Kimodo、Bridge、TextOp 和 MuJoCo 的接口或启动顺序。
- action horizon 仍为 40，执行 34 帧，Prefix-RTC overlap 仍为 6 帧。
- 本 checkpoint 沿用原链路的 `legacy_zero` timestep 语义，prefix bucket 为 0。
- 不修改 checkpoint 的 config、processor 或权重。
- 不直接覆盖已有的导出、运行时或启动脚本。

## 3. 目标 checkpoint 与环境

Checkpoint：

```text
/home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2
```

本机验证环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu128 |
| CUDA | 12.8 |
| TensorRT | 10.15.1.29 |
| ONNX | 1.20.1 |
| GPU | NVIDIA GeForce RTX 4090，SM 8.9 |
| Driver | 595.71.05 |

环境检查：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

.venv/bin/python - <<'PY'
import onnx
import tensorrt as trt
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("onnx", onnx.__version__)
print("tensorrt", trt.__version__)
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("builder", trt.Builder(trt.Logger()) is not None)
PY

uv pip check --python .venv/bin/python
```

本环境不需要 `onnx_graphsurgeon` 或 `polygraphy`；当前导出和构建脚本没有 import 它们。`.venv` 没有内置 pip 时使用 `uv pip` 检查或安装，不要据此判断环境损坏。

## 4. 本机实现与 5880 参考包的关键差异

参考目录：

```text
/home/ubuntu/yzh/gr00t_rtc_trt_pack
```

### 4.1 Adapter 相同

以下两个文件 SHA256 完全一致：

```text
/home/ubuntu/yzh/gr00t_rtc_trt_pack/psi0_adapter/gr00t_n17_prefix_rtc.py
/home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/gr00t_n17_prefix_rtc.py
```

### 4.2 timestep 默认值不同

参考 runtime 默认使用：

```text
groot_clean -> prefix bucket 999
```

本 checkpoint 的原始链路使用：

```text
legacy_zero -> prefix bucket 0
```

不得为了匹配参考 README 而擅自切换到 `groot_clean`。两种语义的离线实测见第 7 节。

### 4.3 导出文件隔离方式不同

参考版会直接将普通 Action Encoder/DiT 导出改造成 Prefix-RTC shape，并继续使用普通文件名。本机实现使用独立分支和独立文件名：

```text
普通 engine:
  action_encoder.engine
  dit_bf16.engine

Prefix-RTC engine:
  action_encoder_prefix_rtc.engine
  dit_prefix_rtc_bf16.engine
```

这样普通 TensorRT 与 Prefix-RTC TensorRT 可以共存，不会互相覆盖。

### 4.4 Runtime 分支不同

参考版使用 `_EngineAsActionEncoder`、`_EngineAsDit`、`_EngineAsActionDecoder` 包装 engine，再调用 adapter 解码循环。

本机版使用独立函数：

```text
action_head_prefix_rtc_tensorrt_forward
```

对应模式为：

```text
prefix_rtc_action_head
```

该路径已经在 `legacy_zero` 下通过同观测、同前序 chunk、同初始噪声对拍。

### 4.5 不使用参考 processor 修改脚本

参考脚本会直接修改 checkpoint 下的 JSON，而且只查找根目录的 `processor_config.json`。本 checkpoint 使用 `processor/processor_config.json`。

本机 `Gr00tPolicy` 支持：

```python
Gr00tPolicy(..., backbone_path="/local/Cosmos-Reason2-2B")
```

因此无需修改 checkpoint metadata。

## 5. Engine contract

安全模式只加载以下四个文件：

| Engine | 输入 | 输出 |
| --- | --- | --- |
| `state_encoder.engine` | state `(1,1,132)`、embodiment `(1,)` | `(1,1,1536)` |
| `action_encoder_prefix_rtc.engine` | action `(1,40,132)`、timestep `(1,40)` | `(1,40,1536)` |
| `dit_prefix_rtc_bf16.engine` | SA `(1,41,1536)`、timestep `(1,41)`、动态 VL sequence | `(1,41,1024)` |
| `action_decoder.engine` | model output `(1,41,1024)` | `(1,41,132)` |

`41 = 1 state token + 40 action tokens`。所有 engine 都是 batch size 1、BF16 action-head contract。

当前已验证文件：

```text
6a6184e98a6d69adc65be2c4f583c342e32e0b4941fb3401e25e5b9c8ead728a  state_encoder.engine
5a8644a41df0764defd648cf187c03ea74da673f0e1fe3ea1fdaa7eaeedba381  action_encoder_prefix_rtc.engine
ced0ca53f51e6c92f4b5879433ebda701c55e4dd7785c2efc8ae48d9fb13b855  dit_prefix_rtc_bf16.engine
7fd0df36de27fae625e3b348649c1453c0436dd1018347dd470e5ca124559f30  action_decoder.engine
```

TensorRT engine 与 GPU 架构和 TensorRT runtime 绑定。更换 GPU capability 或 TensorRT 版本后必须重建和重新对拍。

## 6. 独立检查工具

新增的独立工具：

```text
scripts/deployment/rtc_prefix_action_head_parity_check.py
```

该文件不会被 server、launcher 或 runtime import。它提供：

- 路径与模型资产检查；
- TensorRT engine 反序列化；
- engine IO contract 检查；
- 相同真实观测输入；
- 相同前序 action chunk；
- 相同初始 action noise；
- PyTorch 与 TensorRT normalized action 对拍；
- prefix hard rewrite、cosine、mean/max error 阈值检查。

只检查、不加载模型：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

.venv/bin/python scripts/deployment/rtc_prefix_action_head_parity_check.py \
  --model-path /home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2 \
  --backbone-path /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
  --dataset-path /home/ubuntu/yzh/HumanoidVLA_MJ/output/fullstate_20260615_task1 \
  --engine-dir /home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2/tensorrt/engines \
  --adapter-path /home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/gr00t_n17_prefix_rtc.py \
  --embodiment-tag new_embodiment \
  --overlap 6 \
  --check-only
```

## 7. 离线 Prefix-RTC 对拍

完整命令：

```bash
cd /home/ubuntu/yzh/Isaac-GR00T-rtc

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python scripts/deployment/rtc_prefix_action_head_parity_check.py \
  --model-path /home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2 \
  --backbone-path /home/ubuntu/yzh/Isaac-GR00T-rtc/huggingface/Cosmos-Reason2-2B \
  --dataset-path /home/ubuntu/yzh/HumanoidVLA_MJ/output/fullstate_20260615_task1 \
  --engine-dir /home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2/tensorrt/engines \
  --adapter-path /home/ubuntu/yzh/Psi0_kimodo_textop/scripts/deploy/gr00t_n17_prefix_rtc.py \
  --embodiment-tag new_embodiment \
  --overlap 6 \
  --seed 42 \
  --prefix-timestep-mode legacy_zero \
  --min-cosine 0.99 \
  --max-abs-error 0.1
```

通过结果：

```text
cosine                 0.9999961853
mean absolute error     0.0015824520
max absolute error      0.03125
suffix max abs error    0.03125
PyTorch prefix error    0.0
TensorRT prefix error   0.0
prefix cross error      0.0
```

错误地使用 `groot_clean` 时：

```text
cosine                 0.9992870092
max absolute error      0.248046875
```

虽然 cosine 仍然较高，但 suffix 最大误差明显超限，所以不能只看 cosine，也不能放宽阈值掩盖 timestep 语义错误。

## 8. 默认 PyTorch 启动

不设置 TRT 环境变量即可保持原链路：

```bash
cd /home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ

VLA_BACKEND=gr00t \
GR00T_INFERENCE_BACKEND=pytorch \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2 \
GR00T_PORT=22085 \
GR00T_ACTION_HORIZON=40 \
ACTION_EXEC_HORIZON=34 \
GR00T_RTC=1 \
GR00T_PREFIX_RTC=1 \
bash scripts/deploy/mujoco_psi0_kimodo_textop_commands.sh serve-gr00t
```

`GR00T_INFERENCE_BACKEND=pytorch` 也可以省略，因为 launcher 默认值就是 PyTorch。

## 9. TensorRT 启动

四终端链路中只修改终端 1：

```bash
cd /home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ

VLA_BACKEND=gr00t \
GR00T_INFERENCE_BACKEND=tensorrt \
GR00T_TRT_MODE=prefix_rtc_action_head \
GR00T_TRT_ENGINE_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2/tensorrt/engines \
PREFIX_RTC_TIMESTEP_MODE=legacy_zero \
GR00T_MODEL_PATH=/home/ubuntu/yzh/ckpt/gr00tn17/task3-gr00t-n17-rot6d59-kimodo-textop-prefix-rtc-round2 \
GR00T_PORT=22085 \
GR00T_ACTION_HORIZON=40 \
ACTION_EXEC_HORIZON=34 \
GR00T_RTC=1 \
GR00T_PREFIX_RTC=1 \
bash scripts/deploy/mujoco_psi0_kimodo_textop_commands.sh serve-gr00t
```

成功日志必须包含：

```text
Prefix-RTC action head TRT engines loaded and forward method patched.
Backbone remains in PyTorch (Qwen3-VL).
TensorRT enabled: mode=prefix_rtc_action_head
backend=tensorrt trt_mode=prefix_rtc_action_head
```

变量名是 `PREFIX_RTC_TIMESTEP_MODE`，不是 `GR00T_PREFIX_RTC_TIMESTEP_MODE`。

不要把注释插进以反斜杠续行的命令中，也不要在反斜杠后保留空格，否则部分环境变量不会传入最终命令。

## 10. 独立端口服务检查

服务启动后检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:22086/health
```

本机 shell 设置了 `ALL_PROXY=socks5://127.0.0.1:7890`。不加 `--noproxy '*'` 时 localhost 请求可能被错误送入代理，并表现为 `Empty reply from server`；这不是 GR00T 服务故障。

正确结果应包含：

```json
{
  "status": "ok",
  "inference_backend": "tensorrt",
  "trt_mode": "prefix_rtc_action_head",
  "action_horizon": 40,
  "action_exec_horizon": 34,
  "prefix_rtc": true,
  "rtc_overlap_steps": 6
}
```

## 11. 完整链路验收

启动顺序沿用现有四终端链路。只改变 GR00T 终端，Kimodo、MuJoCo 和 Bridge 指令保持原样。

必须同时满足：

- GR00T 服务显示 `prefix_rtc_action_head`；
- 绿色 Kimodo 重建影子正常；
- 五点约束正常；
- Bridge 持续生成多个 chunk；
- prefix 6 帧续接正常；
- suffix 34 帧没有跳变或发散；
- pause 行为与 PyTorch 基线一致；
- 无 shape、engine、TensorRT 或 timestep 错误。

本 checkpoint 的默认 PyTorch链路、TensorRT独立端口、离线 parity 和完整四终端 TensorRT 链路均已通过。

## 12. 重建 engine

以下情况必须重建：

- 更换 checkpoint；
- 更换 TensorRT runtime；
- 更换 GPU compute capability；
- 修改 action horizon、action dimension、batch size；
- 修改 Prefix-RTC timestep 或 DiT token conditioning；
- engine hash 与本文不一致且来源不明确。

当前 launcher 提供构建入口：

```bash
cd /home/ubuntu/yzh/HumanoidVLA_MJ_backup/HumanoidVLA_MJ

GR00T_MODEL_PATH=/path/to/checkpoint \
GR00T_TRT_OUTPUT_DIR=/path/to/checkpoint/tensorrt \
GR00T_TRT_DATASET_PATH=/path/to/lerobot_dataset \
GR00T_TRT_EMBODIMENT_TAG=new_embodiment \
GR00T_TRT_WORKSPACE_MB=8192 \
bash scripts/deploy/mujoco_psi0_kimodo_textop_commands.sh build-gr00t-trt
```

该入口会导出 `prefix_rtc_full_pipeline` 的全部 ONNX/engine，但生产安全模式只加载四个 action-head engine。构建完成后必须再次运行本文第 6、7、10、11 节验证，不能因为 build 成功就直接投入完整链路。

## 13. 回退

运行时回退只需停止 GR00T 服务，并移除：

```text
GR00T_INFERENCE_BACKEND=tensorrt
GR00T_TRT_MODE=prefix_rtc_action_head
GR00T_TRT_ENGINE_PATH=...
PREFIX_RTC_TIMESTEP_MODE=legacy_zero
```

然后使用第 8 节 PyTorch 命令重启。无需删除 engine、修改 checkpoint 或回退其他三个终端。

如果完全移除本轮新增内容，只需删除独立文件：

```text
scripts/deployment/rtc_prefix_action_head_parity_check.py
scripts/deployment/PREFIX_RTC_ACTION_HEAD_TRT_GUIDE.md
```

不要回退同一 checkout 中原有的 Prefix-RTC、backbone path、TensorRT modes、调试或 launcher 支持代码；这些是本轮开始前已经存在的工作区内容。
