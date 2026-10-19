# 信也科技杯：v161 对话轮次交互建模方案

本仓库整理信也科技杯 2026 复赛的 **v161_bc104** 方案：多模态模型、七源按类别融合、REF top-k 校准，以及 BC / I 后处理规则。

**历史最佳分数：0.668080（macro F1）**，来自项目终局复盘记录。仓库提供对应版本保存的源代码与配置，不包含比赛数据、模型权重、平台账号或私有容器镜像。

## 任务

输入双声道对话音频、文本转写和历史事件标签，预测后续时间窗口中的五类话轮事件：

| 输出列 | 含义 |
|---|---|
| `c` | 继续 |
| `na` | 无动作 |
| `i` | 未完成插话 |
| `bc` | 附和 |
| `t` | 话轮切换 |

模型输出为五列独立预测。训练数据标签与私域评测分布存在差异，最终方案除了学习模型，还利用比赛反馈调整每类输出比例。此校准是比赛数据域上的经验结果，不代表对任意新数据集都适用。

## 最佳方案

### 多模态模型与七源融合

主要模型使用 Whisper-large-v3 音频编码器、Qwen3-0.6B 文本编码器和历史标签特征，结合 LoRA、chunk 辅助监督及不同训练配置形成互补成员；activity TCN 提供额外的声学活动信号。

最佳融合的成员顺序固定为：

```text
main, stride5, textlora, tail25, activity_tcn, s123_heavy, s124_aggressive
```

每个类别使用不同线性权重，配置见 [`fusion_weights_7src_v30_nospecaug.json`](rules/fusion_weights_7src_v30_nospecaug.json)。文件名 v30 指融合配置的来源；最终分数对应 v161 的校准参数，不应将二者误认为同一个版本编号。

### REF top-k 校准

对每一类按融合概率排序，选择 `round(REF[class] * N)` 个样本置 1。v161 固定比例为：

| c | na | i | bc | t |
|---:|---:|---:|---:|---:|
| 0.987 | 0.780 | 0.150 | **0.104** | 0.450 |

核心文件 [`code/tools_fusion_to_csv.py`](code/tools_fusion_to_csv.py) 和保存的版本文件 [`code/sweep/fusion_v161_bc104.py`](code/sweep/fusion_v161_bc104.py) 均保留。后续依次执行 BC 不应期规则和 I overlap 规则。

## 代码导航

| 位置 | 内容 |
|---|---|
| `code/train.py` | 多模态训练入口 |
| `code/infer.py` | 单成员推理 |
| `code/model/` | 多模态、LoRA、损失函数和辅助头 |
| `code/_lib/src/` | 数据加载、底层模型和公共工具 |
| `code/configs/` | 保存的训练配置 |
| `code/tools_activity_tcn_train.py` | activity TCN 训练 |
| `code/tools_activity_tcn_test_infer.py` | activity TCN 推理 |
| `code/tools_fusion_to_csv.py` | 融合与 REF top-k |
| `code/tools_bc_refractory_rule.py` | BC 后处理 |
| `code/tools_i_overlap_rule.py` | I 后处理 |
| `code/tools_acoustic_overlap.py` | overlap 特征提取 |
| `rules/` | 最佳管线需要的三份规则配置 |
| `docs/model-files.json` | 本地保留的七个精简权重的大小与 SHA-256，权重本身不上传 |
| `docs/source-manifest.json` | 导出文件来源和哈希 |

`code/scripts/run_inference.sh` 为随 v161 保存的历史管线，含多个旧版自动选择分支；阅读最佳方案时以本页指定的七源顺序、融合权重和 REF 为准。历史可选分支所需的其他规则或资产没有全部纳入本仓库。

## 融合接口示意

准备同样顺序的七份 `[N, 5]` 概率数组及 ID 列表后，可以参考原接口：

```bash
python code/tools_fusion_to_csv.py \
  --test_probs main.npy stride5.npy textlora.npy tail25.npy activity.npy s123.npy s124.npy \
  --test_ids main_ids.txt stride5_ids.txt textlora_ids.txt tail25_ids.txt activity_ids.txt s123_ids.txt s124_ids.txt \
  --weights_json rules/fusion_weights_7src_v30_nospecaug.json \
  --out_probs fused.npy --out_csv submission_base.csv --out_ids fused_ids.txt
```

这是融合阶段的接口说明，尚需 BC / I 后处理才能组成历史最终管线。它不是无数据、无权重的一键复现命令。

## 成绩口径与整理范围

- 最高分 `0.668080` 引用终局复盘的 v161 记录，本次未重新提交平台评测。
- 不声明最终比赛名次：现存文档的名次记载不一致。本地原文件夹叫 `champion_code_v161`，这只是文件夹命名，不表示本方案获得冠军。
- 以保存的 v161 冻结源码为导出基准，没有将后期重构版本混充原始得分代码。
- 本地保存的七个主成员精简权重合计约 290 MiB；这不包含基座和额外声学模型，也未重新核对它们与当时评分镜像的逐字节对应关系。
- 配置中的个人服务器路径替换为 `/workspace/xinye`；实际使用时需自行调整输入路径。
- `requirements.txt` 是归档依赖参考，并非本次验证的环境锁文件；其中原有 PyTorch 版本注释有差异，应在需要运行时再统一。
- 没有复制队伍私有 Docker 基础镜像配置、平台提交工具、凭证、原始数据、权重和旧 Git 历史。
- 本次仅检查源码语法、文件范围和脱敏情况，不进行训练、推理或成绩复现。

代码来源与许可说明见 [NOTICE.md](NOTICE.md)。
