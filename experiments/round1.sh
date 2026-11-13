#!/bin/bash
# Round 1: which reconstruction changes help? One job list per object (one pod per object).
# All configurations use CLIPasso's loss balance: geometric CLIP conv loss on ViT layers 2,3
# (weight 1) + 0.1 x semantic fc loss, and no extra semantic CLIPLoss (--clip_weight 0).
#   A_repo    the repository's original settings, including its loss (extra CLIPLoss, weight 1): reference
#   B_clipasso A_repo with CLIPasso's loss balance
#   C_fixed   B + 128 spheres per curve, batched (speed; should not change quality)
#   D_bw      C + black strokes on white
#   E_hull    C + visual-hull initialization (longer initial strokes)
#   F_mv      C + 4 views per step, 1 step per batch
#   G_all     C + D + E + F
# Strokes: thickness 0.008 (sigma ~0.75 px at 224 px, like CLIPasso's 1.5 px strokes).
# Usage (bash >= 4, i.e. on the pods): experiments/round1.sh [objects...]   (default: 6 diverse objects)
set -euo pipefail
cd "$(dirname "$0")/.."
OBJECTS=("$@"); [ ${#OBJECTS[@]} -gt 0 ] || OBJECTS=("car_(automobile)" airplane chair teapot bicycle guitar)
# offline on the training pods; the CPU pod live-syncs runs/ to wandb (pod scripts: wandb_sync.sh)
WANDB=(--wandb --wandb_mode offline --wandb_entity efe-tarhan-epfl --wandb_project clipasso3d)
COMMON=(--n_curves 25 --thickness 0.008 --gif_fps 30 --rotation_time 6 "${WANDB[@]}")
CLIPASSO=(--clip_weight 0 --clip_conv_loss 1 --clip_fc_loss_weight 0.1 --clip_conv_layer_weights 0 0 1 1 0)
OLD=(--epochs 10 --batch_size 1 --inner_steps 30 --frame_every 5)
MV=(--epochs 120 --batch_size 4 --inner_steps 1 --frame_every 15)
FIX=(--samples_per_curve 128)
HULL=(--init hull --length 0.15)
BW=(--style black_on_white)

declare -A CFG=(
  [A_repo]="${OLD[*]}"
  [B_clipasso]="${OLD[*]} ${CLIPASSO[*]}"
  [C_fixed]="${OLD[*]} ${CLIPASSO[*]} ${FIX[*]}"
  [D_bw]="${OLD[*]} ${CLIPASSO[*]} ${FIX[*]} ${BW[*]}"
  [E_hull]="${OLD[*]} ${CLIPASSO[*]} ${FIX[*]} ${HULL[*]}"
  [F_mv]="${MV[*]} ${CLIPASSO[*]} ${FIX[*]}"
  [G_all]="${MV[*]} ${CLIPASSO[*]} ${FIX[*]} ${HULL[*]} ${BW[*]}"
)
NAMES=(B_clipasso C_fixed D_bw E_hull F_mv G_all A_repo)   # CLIPasso-loss runs first
mkdir -p jobs/round1
for obj in "${OBJECTS[@]}"; do
  out="jobs/round1/${obj//[()]/}.jsonl"; rm -f "$out"   # no parentheses in file names (shell-safe pod commands)
  for name in "${NAMES[@]}"; do
    # shellcheck disable=SC2086
    python tools/sweep.py make --datasets "data/objaverse/$obj" --run_name "round1_$name" \
      --runs_root runs --out "$out" --append -- "${COMMON[@]}" ${CFG[$name]} >/dev/null
  done
  echo "$out: $(wc -l < "$out") jobs"
done
