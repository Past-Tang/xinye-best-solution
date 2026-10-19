#!/bin/bash
# 复赛 Docker 一键推理脚本：5 个成员 infer → 融合 → BC/I 规则 → submission.csv
#
# Usage:
#   bash code/scripts/run_inference.sh <PKG_ROOT> <DATA_ROOT> <OUTPUT_ROOT>
#
# 前置：已运行 patch_paths.sh 调整 yaml 路径；XINYE_WHISPER / XINYE_QWEN env 已 export
#
# 输出：$OUTPUT_ROOT/submission.csv

set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <PKG_ROOT> <DATA_ROOT> <OUTPUT_ROOT>"
  echo "  PKG_ROOT     rematch_pkg 根 (含 code/ models/ rules/)"
  echo "  DATA_ROOT    数据根 (含 test/audio test/context 等)"
  echo "  OUTPUT_ROOT  本次推理输出目录 (用于临时 npy/csv)"
  exit 1
fi

PKG="${1%/}"; DATA="${2%/}"; OUT="${3%/}"
[ -d "$PKG/code" ] && [ -d "$PKG/models/slim_ckpts" ] || \
  { echo "ERROR: PKG_ROOT 看起来不对 ($PKG)"; exit 1; }

mkdir -p "$OUT/probs"
PY="${PY:-python}"

declare -A MAP=(
  [main]="lora_chunkaux_5gpu.yaml main_e12_slim.pt"
  [stride5]="lora_chunkaux_stride5_5gpu.yaml stride5_e16_slim.pt"
  [textlora]="lora_textlora_4gpu.yaml textlora_e09_slim.pt"
  [s123]="lora_seed123_heavy_5gpu.yaml s123_heavy_slim.pt"
  [tail25]="lora_temporal_tail25_5gpu.yaml tail25_slim.pt"
  [specaug]="lora_specaug_5gpu.yaml specaug_avg_slim.pt"
  [s124]="lora_seed124_aggressive_5gpu.yaml s124_aggressive_slim.pt"
  [wavlm]="lora_wavlm_5gpu_fix.yaml wavlm_fix_slim.pt"
  [hubert]="lora_chinese_hubert_5gpu.yaml hubert_zh_slim.pt"
)

echo "[run] Step 1/6: 多个 LoRA 成员独立推理 (whisper+qwen 共享 base)"
# Auto-detect available members
LORA_LIST="main stride5 textlora s123 tail25"
[ -f "$PKG/models/slim_ckpts/specaug_avg_slim.pt" ] && LORA_LIST="$LORA_LIST specaug"
[ -f "$PKG/models/slim_ckpts/s124_aggressive_slim.pt" ] && LORA_LIST="$LORA_LIST s124"
[ -f "$PKG/models/slim_ckpts/wavlm_fix_slim.pt" ] && [ -d "$PKG/models/wavlm-large" ] && LORA_LIST="$LORA_LIST wavlm"
[ -f "$PKG/models/slim_ckpts/hubert_zh_slim.pt" ] && [ -d "$PKG/models/chinese-hubert-large" ] && LORA_LIST="$LORA_LIST hubert"
for m in $LORA_LIST; do
  read cfg ck <<< "${MAP[$m]}"
  if [ -f "$OUT/probs/${m}_test.npy" ]; then
    echo "  [$m] skip (exists)"
    continue
  fi
  echo "  [$m] cfg=$cfg ckpt=$ck"
  $PY "$PKG/code/infer.py" \
    --config "$PKG/code/configs/$cfg" \
    --checkpoint "$PKG/models/slim_ckpts/$ck" \
    --test_root "$DATA/test" \
    --output_probs "$OUT/probs/${m}_test.npy" \
    --output_ids   "$OUT/probs/${m}_test_ids.txt" \
    --device "${CUDA_DEVICE:-cuda:0}"
done

echo "[run] Step 2/6: activity_tcn 独立 TCN 推理 (轻量 3.8M，正交声学活动信号)"
if [ -f "$OUT/probs/activity_tcn_test.npy" ]; then
  echo "  skip (exists)"
else
  $PY "$PKG/code/tools_activity_tcn_test_infer.py" \
    --checkpoint "$PKG/models/slim_ckpts/activity_tcn_best.pt" \
    --test_root "$DATA/test" \
    --device "${CUDA_DEVICE:-cuda:0}"
  # 拷贝到约定路径（activity_tcn 默认写 outputs/test_probs/）
  mv -f "$PKG/code/outputs/test_probs/activity_tcn_test.npy" "$OUT/probs/" 2>/dev/null \
    || mv -f outputs/test_probs/activity_tcn_test.npy "$OUT/probs/" 2>/dev/null || true
  mv -f "$PKG/code/outputs/test_probs/activity_tcn_test_ids.txt" "$OUT/probs/" 2>/dev/null \
    || mv -f outputs/test_probs/activity_tcn_test_ids.txt "$OUT/probs/" 2>/dev/null || true
fi

# t_boundary_gbdt 产物：优先用预打包的；否则可重训生成（参见 README "重训" 章节）
TB_TEST="$PKG/models/precomputed_test_probs/t_boundary_gbdt.npy"
TB_TEST_IDS="$PKG/models/precomputed_test_probs/t_boundary_gbdt_ids.txt"
if [ -f "$TB_TEST" ] && [ -f "$TB_TEST_IDS" ]; then
  USE_6SRC=1
  cp -n "$TB_TEST" "$OUT/probs/" 2>/dev/null || true
  cp -n "$TB_TEST_IDS" "$OUT/probs/" 2>/dev/null || true
  echo "[run] Step 3/6: 6src 融合（4 LoRA + activity_tcn + t_boundary_gbdt 预打包，权重 fusion_weights_6src.json → valid macroF1 0.6420 ref_rate_f1 obj）"
  $PY "$PKG/code/tools_fusion_to_csv.py" \
    --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                 "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                 "$OUT/probs/activity_tcn_test.npy" \
                 "$OUT/probs/t_boundary_gbdt.npy" \
    --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                 "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                 "$OUT/probs/activity_tcn_test_ids.txt" \
                 "$OUT/probs/t_boundary_gbdt_ids.txt" \
    --weights_json "$PKG/rules/fusion_weights_6src.json" \
    --out_probs "$OUT/probs/fused.npy" \
    --out_csv   "$OUT/submission_base.csv" \
    --out_ids   "$OUT/probs/fused_ids.txt"
else
  # v25: 在 fusion 之前插 Step 2.5 = semantic-turn-taking 推理（如果模型存在）
  STT_MODEL="$PKG/models/semantic_turn_taking"
  STT_PROBS="$OUT/probs/semantic_turn_test.npy"
  STT_IDS="$OUT/probs/semantic_turn_test_ids.txt"
  if [ -d "$STT_MODEL" ] && [ ! -f "$STT_PROBS" ]; then
    echo "[run] Step 2.5/6: semantic-turn-taking (0.5B Qwen2.5 fine-tune) infer"
    $PY "$PKG/code/tools_semantic_turn_test_infer.py" \
      --model_path "$STT_MODEL" \
      --text_dir "$DATA/test/text" \
      --output_probs "$STT_PROBS" \
      --output_ids "$STT_IDS" \
      --device "${CUDA_DEVICE:-cuda:0}"
  fi

  if [ -f "$STT_PROBS" ] && [ -f "$STT_IDS" ]; then
    echo "[run] Step 3/6: 7src_v25 融合（6src + semantic_turn）"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$STT_PROBS" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$STT_IDS" \
      --weights_json "$PKG/rules/fusion_weights_7src_v25.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  elif [ -f "$OUT/probs/hubert_test.npy" ] && [ ! -f "$OUT/probs/specaug_test.npy" ] && [ ! -f "$OUT/probs/wavlm_test.npy" ] && [ -f "$PKG/rules/fusion_weights_8src_v50.json" ]; then
    echo "[run] Step 3/6: 8src_v50 融合 (v30 base + chinese-hubert 跨家族 8th src)"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$OUT/probs/s124_test.npy" \
                   "$OUT/probs/hubert_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$OUT/probs/s124_test_ids.txt" \
                   "$OUT/probs/hubert_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_8src_v50.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  elif [ -f "$OUT/probs/wavlm_test.npy" ] && [ ! -f "$OUT/probs/specaug_test.npy" ] && [ -f "$PKG/rules/fusion_weights_8src_v33.json" ]; then
    echo "[run] Step 3/6: 8src_v33 融合 (v30 base + wavlm, 跳过 specaug)"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$OUT/probs/s124_test.npy" \
                   "$OUT/probs/wavlm_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$OUT/probs/s124_test_ids.txt" \
                   "$OUT/probs/wavlm_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_8src_v33.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  elif [ -f "$OUT/probs/wavlm_test.npy" ] && [ -f "$PKG/rules/fusion_weights_9src_v29.json" ]; then
    echo "[run] Step 3/6: 9src_v29 融合 (8src + wavlm)"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$OUT/probs/specaug_test.npy" \
                   "$OUT/probs/s124_test.npy" \
                   "$OUT/probs/wavlm_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$OUT/probs/specaug_test_ids.txt" \
                   "$OUT/probs/s124_test_ids.txt" \
                   "$OUT/probs/wavlm_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_9src_v29.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  elif [ -f "$OUT/probs/s124_test.npy" ] && [ ! -f "$OUT/probs/specaug_test.npy" ] && [ -f "$PKG/rules/fusion_weights_7src_v30_nospecaug.json" ]; then
    echo "[run] Step 3/6: 7src_v30 融合 (v26 base 6src + s124, 跳过 specaug 减 -0.003 拖累)"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$OUT/probs/s124_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$OUT/probs/s124_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_7src_v30_nospecaug.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  elif [ -f "$OUT/probs/s124_test.npy" ] && [ -f "$PKG/rules/fusion_weights_8src_v28.json" ]; then
    echo "[run] Step 3/6: 8src_v28 融合 (7src + s124_aggressive I×7 BC×4)"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$OUT/probs/specaug_test.npy" \
                   "$OUT/probs/s124_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$OUT/probs/specaug_test_ids.txt" \
                   "$OUT/probs/s124_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_8src_v28.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  elif [ -f "$OUT/probs/specaug_test.npy" ] && [ -f "$PKG/rules/fusion_weights_7src_v27.json" ]; then
    echo "[run] Step 3/6: 7src_v27 融合（6 LoRA + activity_tcn + s123 + specaug_avg）"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
                   "$OUT/probs/specaug_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
                   "$OUT/probs/specaug_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_7src_v27.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  else
    echo "[run] Step 3/6: 6src_v21 fallback (无 specaug)"
    $PY "$PKG/code/tools_fusion_to_csv.py" \
      --test_probs "$OUT/probs/main_test.npy" "$OUT/probs/stride5_test.npy" \
                   "$OUT/probs/textlora_test.npy" "$OUT/probs/tail25_test.npy" \
                   "$OUT/probs/activity_tcn_test.npy" \
                   "$OUT/probs/s123_test.npy" \
      --test_ids   "$OUT/probs/main_test_ids.txt" "$OUT/probs/stride5_test_ids.txt" \
                   "$OUT/probs/textlora_test_ids.txt" "$OUT/probs/tail25_test_ids.txt" \
                   "$OUT/probs/activity_tcn_test_ids.txt" \
                   "$OUT/probs/s123_test_ids.txt" \
      --weights_json "$PKG/rules/fusion_weights_6src_v21.json" \
      --out_probs "$OUT/probs/fused.npy" \
      --out_csv   "$OUT/submission_base.csv" \
      --out_ids   "$OUT/probs/fused_ids.txt"
  fi
fi

echo "[run] Step 4/6: 抽取声学 overlap"
CALLHOME=$(ls "$PKG/models/callhome-zho/snapshots/"*/model.safetensors 2>/dev/null | head -1)
[ -n "$CALLHOME" ] || { echo "ERROR: callhome-zho safetensors 未找到"; exit 1; }
if [ -f "$OUT/probs/acoustic_overlap_test.npy" ]; then
  echo "  skip (exists)"
else
  $PY "$PKG/code/tools_acoustic_overlap.py" \
    --audio "$DATA/test/audio" \
    --ckpt "$CALLHOME" \
    --out "$OUT/probs/acoustic_overlap_test.npy" \
    --device "${CUDA_DEVICE:-cuda:0}"
fi

echo "[run] Step 5/6: BC 不应期规则 (删除 BC FP)"
$PY "$PKG/code/tools_bc_refractory_rule.py" apply \
  --rule "$PKG/rules/bc_refractory_rule.json" \
  --probs "$OUT/probs/fused.npy" \
  --ids "$OUT/probs/fused_ids.txt" \
  --base_csv "$OUT/submission_base.csv" \
  --out "$OUT/submission_bc.csv" \
  --test_context_dir "$DATA/test/context"

echo "[run] Step 6/6: I 声学 overlap 规则 (删除 I FP)"
$PY "$PKG/code/tools_i_overlap_rule.py" apply \
  --rule "$PKG/rules/i_overlap_rule.json" \
  --probs "$OUT/probs/fused.npy" \
  --ids "$OUT/probs/fused_ids.txt" \
  --base_csv "$OUT/submission_bc.csv" \
  --out "$OUT/submission.csv" \
  --acoustic_overlap_probs "$OUT/probs/acoustic_overlap_test.npy" \
  --acoustic_overlap_ids "$OUT/probs/acoustic_overlap_test_ids.txt"

echo "[run] 完成。最终提交文件：$OUT/submission.csv"
$PY - <<EOF
import csv
rows = list(csv.DictReader(open("$OUT/submission.csv")))
for c in ["c","na","i","bc","t"]:
    print(f"  {c}={sum(1 for r in rows if r[c]=='1')}")
print(f"  total rows = {len(rows)}")
EOF
