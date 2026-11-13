#!/bin/bash
# Main ablations: number of lines x abstraction, one job list per object (one pod per object).
# Thickness is fixed (0.008). The base configuration is the winner of round 1 (override with BASE=...).
#
# Abstraction = balance of CLIPasso's two losses (no extra semantic CLIPLoss, --clip_weight 0):
#   geometric: L2 between CLIP ViT layers 2,3 of sketch and target (per-layer weights below)
#   semantic:  cosine distance of the final CLIP embeddings (--clip_fc_loss_weight)
#   exact          geometric 1.0, semantic 0      strokes follow the contours
#   normal         geometric 1.0, semantic 0.1    CLIPasso default
#   abstract       geometric 1.0, semantic 1.0
#   very_abstract  geometric 0.1, semantic 1.0
#
# Usage (bash >= 4, on the pods): experiments/ablation.sh [objects...]   (default: all of data/objaverse)
set -euo pipefail
cd "$(dirname "$0")/.."
OBJECTS=("$@")
if [ ${#OBJECTS[@]} -eq 0 ]; then
  for d in data/objaverse/*/; do [ -f "$d/transforms_test.json" ] && OBJECTS+=("$(basename "$d")"); done
fi
WANDB=(--wandb --wandb_mode offline --wandb_entity efe-tarhan-epfl --wandb_project clipasso3d)
BASE=${BASE:-"--epochs 120 --batch_size 4 --inner_steps 1 --frame_every 15 --samples_per_curve 128 --init hull --length 0.15 --style black_on_white"}
COMMON=(--thickness 0.008 --rasterizer sparse --clip_weight 0 --gif_fps 30 --rotation_time 6 "${WANDB[@]}")
LINES=(10 20 50 100 200 500 1000)
declare -A ABS=(
  [exact]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 0"
  [normal]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 0.1"
  [abstract]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 1.0"
  [very_abstract]="--clip_conv_layer_weights 0 0 0.1 0.1 0 --clip_fc_loss_weight 1.0"
)
mkdir -p jobs/ablation
for obj in "${OBJECTS[@]}"; do
  out="jobs/ablation/${obj//[()]/}.jsonl"; rm -f "$out"
  for level in normal exact abstract very_abstract; do
    for n in "${LINES[@]}"; do
      # shellcheck disable=SC2086
      python tools/sweep.py make --datasets "data/objaverse/$obj" --run_name "abl_L${n}_${level}" \
        --runs_root runs --out "$out" --append -- "${COMMON[@]}" $BASE --n_curves "$n" ${ABS[$level]} >/dev/null
    done
  done
  echo "$out: $(wc -l < "$out") jobs"
done
