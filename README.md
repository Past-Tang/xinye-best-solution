# 信也科技杯：多模态对话轮次预测

本方案用于信也科技杯 2026 复赛，通过对话音频、文本转写和历史事件标签，预测未来时间窗口中的话轮事件。最终采用 **v161_bc104** 配置，复赛最佳成绩为 **0.668080 macro F1**。

## 任务与模型

模型为每个对话片段输出五个独立概率，分别对应继续（c）、无动作（na）、未完成插话（i）、附和（bc）和话轮切换（t）。

音频部分使用 Whisper-large-v3 编码双声道语音，文本部分使用 Qwen3-0.6B，历史事件标签提供对话状态信息。音频和文本编码器通过 LoRA 微调，融合后的特征用于预测五类事件。训练中加入 chunk 级辅助监督，帮助模型学习片段内的事件变化。

最终使用六个多模态成员和一个 activity TCN。多模态成员通过帧步长、文本 LoRA、尾部建模及类别权重等训练设置形成差异；activity TCN 提供声学活动特征。各成员独立推理，再按类别融合概率。

## 七源融合

五个类别分别使用一组权重。这样可以让同一个成员在不同类别中承担不同的预测比重。成员顺序及权重如下。

| 成员 | c | na | i | bc | t |
|---|---:|---:|---:|---:|---:|
| main | 0.16 | 0.00 | 0.05 | 0.07 | 0.20 |
| stride5 | 0.50 | 0.20 | 0.20 | 0.10 | 0.50 |
| textlora | 0.00 | 0.00 | 0.10 | 0.10 | 0.00 |
| tail25 | 0.00 | 0.36 | 0.00 | 0.05 | 0.00 |
| activity_tcn | 0.14 | 0.14 | 0.05 | 0.04 | 0.10 |
| s123_heavy | 0.10 | 0.20 | 0.25 | 0.32 | 0.10 |
| s124_aggressive | 0.10 | 0.10 | 0.35 | 0.32 | 0.10 |

每列权重之和为 1，融合概率为各成员概率的加权和。`s123_heavy` 和 `s124_aggressive` 在 i、bc 两类中的权重较高，`stride5` 则在 c、t 两类中占较大比重。

权重保存在 [融合配置](rules/fusion_weights_7src_v30_nospecaug.json)，实现位于 [tools_fusion_to_csv.py](code/tools_fusion_to_csv.py)。

## REF top-k 校准

训练标签与私域评测标签的分布存在差异，最终方案按类别控制预测为正的样本数量。对每一列融合概率降序排列，取前 `round(REF × N)` 个样本置为 1，其余置为 0。

| 类别 | c | na | i | bc | t |
|---|---:|---:|---:|---:|---:|
| REF | 0.987 | 0.780 | 0.150 | **0.104** | 0.450 |

这些比例根据比赛反馈调整，适用于本次评测的数据分布。以 bc 为例，校准阶段选取约 10.4% 的样本作为附和事件候选，随后再通过后处理规则删除部分候选。

## BC 与 I 后处理

BC 规则结合历史附和比例和模型置信度。如果历史上下文中的 BC 比例较高，而当前预测的置信度较低，就将该样本的 BC 预测改为 0。

I 规则使用片段末尾 5 秒的语音重叠特征。如果语音重叠较少，且模型对插话的预测置信度较低，就将 I 预测改为 0。语音重叠特征由额外的声学模型提取。

两条规则均只删除符合条件的正例预测。完整处理顺序为：

```text
七个成员独立推理
    → 按类别加权融合
    → REF top-k 校准
    → BC 不应期规则
    → I overlap 规则
    → 五列事件预测
```

规则参数分别位于 [bc_refractory_rule.json](rules/bc_refractory_rule.json) 和 [i_overlap_rule.json](rules/i_overlap_rule.json)。

## 代码结构

| 位置 | 功能 |
|---|---|
| `code/train.py` | 多模态成员训练 |
| `code/infer.py` | 单成员推理 |
| `code/model/` | 模型、LoRA、损失函数与辅助头 |
| `code/_lib/src/` | 数据加载与公共工具 |
| `code/configs/` | 各成员训练配置 |
| `code/tools_activity_tcn_train.py` | activity TCN 训练 |
| `code/tools_activity_tcn_test_infer.py` | activity TCN 推理 |
| `code/tools_fusion_to_csv.py` | 七源融合与 top-k 校准 |
| `code/tools_acoustic_overlap.py` | 提取语音重叠特征 |
| `code/tools_bc_refractory_rule.py` | BC 后处理 |
| `code/tools_i_overlap_rule.py` | I 后处理 |

## 融合调用

输入为七份 `[N, 5]` 概率数组和对应的样本 ID 文件，成员顺序与权重表一致。

```bash
python code/tools_fusion_to_csv.py \
  --test_probs main.npy stride5.npy textlora.npy tail25.npy activity.npy s123.npy s124.npy \
  --test_ids main_ids.txt stride5_ids.txt textlora_ids.txt tail25_ids.txt activity_ids.txt s123_ids.txt s124_ids.txt \
  --weights_json rules/fusion_weights_7src_v30_nospecaug.json \
  --out_probs fused.npy \
  --out_csv submission_base.csv \
  --out_ids fused_ids.txt
```

`submission_base.csv` 是 top-k 校准后的结果，之后依次执行 BC 和 I 规则得到最终预测。数据与模型权重需另行准备，七个成员的权重文件清单见 [model-files.json](docs/model-files.json)。
