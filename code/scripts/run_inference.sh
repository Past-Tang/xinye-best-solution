#!/usr/bin/env bash
# Six multimodal members + Activity TCN, followed by fusion and BC/I rules.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <REPO_ROOT> <DATA_ROOT> <NEW_OUTPUT_DIR>" >&2
  exit 2
fi
PKG="$(cd -- "$1" && pwd)"
DATA="$(cd -- "$2" && pwd)"
PY="${PY:-python}"
MODELS="${MODEL_ROOT:-$PKG/models}"
MODELS="$(cd -- "$MODELS" && pwd)"
CKPTS="${SLIM_CKPT_ROOT:-$MODELS/slim_ckpts}"
for d in audio text context; do
  [ -d "$DATA/test/$d" ] || { echo "Missing test/$d under DATA_ROOT" >&2; exit 1; }
done
for d in whisper-large-v3 Qwen3-0.6B; do
  [ -d "$MODELS/$d" ] || { echo "Missing base model: $MODELS/$d" >&2; exit 1; }
done
declare -A CONFIG=(
  [main]=lora_chunkaux_5gpu.yaml [stride5]=lora_chunkaux_stride5_5gpu.yaml
  [textlora]=lora_textlora_4gpu.yaml [tail25]=lora_temporal_tail25_5gpu.yaml
  [s123]=lora_seed123_heavy_5gpu.yaml [s124]=lora_seed124_aggressive_5gpu.yaml
)
declare -A WEIGHT=(
  [main]=main_e12_slim.pt [stride5]=stride5_e16_slim.pt
  [textlora]=textlora_e09_slim.pt [tail25]=tail25_slim.pt
  [s123]=s123_heavy_slim.pt [s124]=s124_aggressive_slim.pt
)
for name in main stride5 textlora tail25 s123 s124; do
  [ -f "$CKPTS/${WEIGHT[$name]}" ] || { echo "Missing member: $name" >&2; exit 1; }
done
[ -f "$CKPTS/activity_tcn_best.pt" ] || { echo 'Missing Activity TCN' >&2; exit 1; }
if [ -z "${CALLHOME_CKPT:-}" ]; then
  shopt -s nullglob
  candidates=("$MODELS/callhome-zho/model.safetensors" "$MODELS"/callhome-zho/snapshots/*/model.safetensors)
  existing=()
  for p in "${candidates[@]}"; do [ ! -f "$p" ] || existing+=("$p"); done
  [ "${#existing[@]}" -eq 1 ] || { echo 'Set CALLHOME_CKPT to one segmentation checkpoint' >&2; exit 1; }
  CALLHOME_CKPT="${existing[0]}"
fi
[ -f "$CALLHOME_CKPT" ] || { echo 'CALLHOME_CKPT does not exist' >&2; exit 1; }
# A fresh directory prevents stale probabilities from another test set being reused.
mkdir -- "$3"
OUT="$(cd -- "$3" && pwd)"
mkdir -p "$OUT/probs" "$OUT/configs"
export XINYE_WHISPER="$MODELS/whisper-large-v3" XINYE_QWEN="$MODELS/Qwen3-0.6B"
for name in main stride5 textlora tail25 s123 s124; do
  "$PY" "$PKG/code/prepare_runtime_config.py" \
    --input "$PKG/code/configs/${CONFIG[$name]}" --output "$OUT/configs/$name.yaml" \
    --code-root "$PKG/code" --data-root "$DATA" --run-root "$OUT/$name" \
    --model-root "$MODELS" --cache-root "$OUT/.cache"
  "$PY" "$PKG/code/infer.py" --config "$OUT/configs/$name.yaml" \
    --checkpoint "$CKPTS/${WEIGHT[$name]}" --test_root "$DATA/test" \
    --output_probs "$OUT/probs/${name}_test.npy" --output_ids "$OUT/probs/${name}_test_ids.txt" \
    --device "${CUDA_DEVICE:-cuda:0}"
done
"$PY" "$PKG/code/tools_activity_tcn_test_infer.py" \
  --checkpoint "$CKPTS/activity_tcn_best.pt" --test_root "$DATA/test" \
  --config "$PKG/code/configs/sota_e12_ckpt_config.yaml" \
  --output_probs "$OUT/probs/activity_tcn_test.npy" --output_ids "$OUT/probs/activity_tcn_test_ids.txt" \
  --device "${CUDA_DEVICE:-cuda:0}"
probs=(); ids=()
for name in main stride5 textlora tail25 activity_tcn s123 s124; do
  probs+=("$OUT/probs/${name}_test.npy"); ids+=("$OUT/probs/${name}_test_ids.txt")
done
"$PY" "$PKG/code/tools_fusion_to_csv.py" --test_probs "${probs[@]}" --test_ids "${ids[@]}" \
  --weights_json "$PKG/rules/fusion_weights_7src_v30_nospecaug.json" \
  --out_probs "$OUT/probs/fused.npy" --out_csv "$OUT/submission_base.csv" --out_ids "$OUT/probs/fused_ids.txt"
"$PY" "$PKG/code/tools_acoustic_overlap.py" --audio "$DATA/test/audio" \
  --ckpt "$CALLHOME_CKPT" --out "$OUT/probs/acoustic_overlap_test.npy" --device "${CUDA_DEVICE:-cuda:0}"
"$PY" "$PKG/code/tools_bc_refractory_rule.py" apply --rule "$PKG/rules/bc_refractory_rule.json" \
  --probs "$OUT/probs/fused.npy" --ids "$OUT/probs/fused_ids.txt" \
  --base_csv "$OUT/submission_base.csv" --out "$OUT/submission_bc.csv" --test_context_dir "$DATA/test/context"
"$PY" "$PKG/code/tools_i_overlap_rule.py" apply --rule "$PKG/rules/i_overlap_rule.json" \
  --probs "$OUT/probs/fused.npy" --ids "$OUT/probs/fused_ids.txt" \
  --base_csv "$OUT/submission_bc.csv" --out "$OUT/submission.csv" \
  --acoustic_overlap_probs "$OUT/probs/acoustic_overlap_test.npy" --acoustic_overlap_ids "$OUT/probs/acoustic_overlap_test_ids.txt"
echo "Saved: $OUT/submission.csv"
