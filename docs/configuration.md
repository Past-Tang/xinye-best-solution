# 成员配置与运行入口

六个多模态成员的 YAML 从保存的 checkpoint 配置整理而来，保留模型和训练参数，将数据、模型与输出位置改为示例路径。对应来源见 [training-config-sources.json](training-config-sources.json)。

| 成员 | 配置 | 结构要点 |
|---|---|---|
| main | `lora_chunkaux_5gpu.yaml` | 双声道均值，ChunkAux，不使用 FutureAux |
| stride5 | `lora_chunkaux_stride5_5gpu.yaml` | 均值与绝对差，步长 5，不使用 FutureAux |
| textlora | `lora_textlora_4gpu.yaml` | 文本 LoRA、FutureAux、BC 辅助损失 |
| tail25 | `lora_temporal_tail25_5gpu.yaml` | 逐帧时序结构，末尾 25 帧 query，FutureAux |
| s123 | `lora_seed123_heavy_5gpu.yaml` | q/v LoRA 与类别正例加权 |
| s124 | `lora_seed124_aggressive_5gpu.yaml` | q/v LoRA 与另一组类别正例权重 |
| activity_tcn | `activity_tcn.json` | 活动特征、6 层 TCN；JSON 记录训练命令参数 |

Activity TCN 的特征提取依赖 `tools_audio_state_quick_oof.py`。文件中保留了音频活动统计、文本活动特征、采样窗口和指标计算，训练与推理使用同一套特征定义。

## 配置路径

在仓库根目录执行。以下命令只生成配置，数据和基础模型需自行提供：

```bash
python code/prepare_runtime_config.py \
  --input code/configs/lora_chunkaux_5gpu.yaml \
  --output outputs/main.yaml \
  --code-root code --data-root /path/to/data \
  --run-root outputs/main --model-root /path/to/models \
  --cache-root .cache
```

`models` 下使用 `whisper-large-v3`、`Qwen3-0.6B` 两个基础模型目录。训练入口为：

```bash
python code/train.py --config outputs/main.yaml
# 多 GPU 时在已激活的环境中使用 torchrun：
torchrun --standalone --nproc_per_node=2 code/train.py --config outputs/main.yaml
```

Activity TCN 训练示例中的参数来自保存配置：

```bash
python code/tools_activity_tcn_train.py \
  --config "$PWD/outputs/main.yaml" \
  --output_root "$PWD/outputs/activity_tcn" \
  --out "$PWD/outputs/activity_tcn/oof.npz" \
  --epochs 18 --batch_size 1024 --eval_batch_size 4096 \
  --hidden 128 --layers 6 --dropout 0.15 --stride 25 \
  --horizons 5,10,15,25 --device cuda
```

## 七成员推理

`code/scripts/run_inference.sh` 固定使用六个多模态成员与 Activity TCN，按 `main / stride5 / textlora / tail25 / activity_tcn / s123 / s124` 的顺序融合，然后应用 BC 和 I 规则。输出目录必须尚不存在，避免误用另一份测试集留下的概率文件。

```bash
MODEL_ROOT=/path/to/models \
SLIM_CKPT_ROOT=/path/to/member_checkpoints \
CALLHOME_CKPT=/path/to/callhome/model.safetensors \
bash code/scripts/run_inference.sh "$PWD" /path/to/data /path/to/new_output
```

七个成员缺少任意一个时，脚本会在推理前退出。加载 slim checkpoint 时只允许省略文件中明确记录的基础模型参数；缺少任务投影、LoRA 或分类头参数会报错，避免以随机初始化参数继续推理。

当前仓库提供代码、配置和训练参数记录，不分发基础模型、成员权重或比赛数据。验证覆盖配置一致性、关键函数和合成输入上的管线；没有重新运行完整训练或官方评测。
