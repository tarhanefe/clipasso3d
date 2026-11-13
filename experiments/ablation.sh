#!/bin/bash
# Main ablation: number of lines x abstraction level, on every rendered object.
# One shared job list (jobs/ablation/all.jsonl) that any number of pods/GPUs/workers work through.
#
# Fixed for all runs (the run.py defaults): thickness 0.008, black strokes on white, visual-hull
# initialization, 8 different views per step, all curves sampled at once with adaptive density,
# no connectivity constraint. From batch 490 on, every 195 batches, curves that ended up off the object
# (outside the visual hull) or collapsed are respawned as new short lines on the least covered part of
# the hull, so every run keeps exactly its number of lines (early stopping restarts after a respawn).
# Each run stops when the validation loss (every 100 batches) has not improved by 0.5% for 5
# validations, after at least 500 and at most 7800 batches (600 epochs).
#
# Abstraction = balance of CLIPasso's two losses (no extra semantic CLIPLoss, --clip_weight 0):
#   geometric: L2 between CLIP ViT layers 2,3 of sketch and target (per-layer weights)
#   semantic:  cosine distance of the final CLIP embeddings (--clip_fc_loss_weight)
#                       geometric  semantic  semantic/geometric
#   too_geometric          1.0       0          0
#   middle_geometric       1.0       0.02       0.02
#   normal                 1.0       0.1        0.1     (CLIPasso)
#   middle_abstract        1.0       1.0        1
#   too_abstract           0.1       1.0        10
#
# Usage (bash >= 4, on the pods): experiments/ablation.sh [objects...]   (default: all of data/objaverse)
set -euo pipefail
cd "$(dirname "$0")/.."
OBJECTS=("$@")
if [ ${#OBJECTS[@]} -eq 0 ]; then
  for d in data/objaverse/*/; do [ -f "$d/transforms_test.json" ] && OBJECTS+=("$(basename "$d")"); done
fi
WANDB=(--wandb --wandb_mode offline --wandb_entity efe-tarhan-epfl --wandb_project clipasso3d)
STOP=(--epochs 600 --eval_interval 100 --early_stop_patience 5 --early_stop_tol 0.005 --min_batches 500)
PRUNE=(--prune_mode respawn --prune_start 490 --prune_every 195)
COMMON=(--thickness 0.008 --snapshot_every 20 --frame_every 10 --gif_fps 30 --rotation_time 6
        "${STOP[@]}" "${PRUNE[@]}" "${WANDB[@]}")
LINES=(5 10 20 50 100 200)
LEVELS=(too_geometric middle_geometric normal middle_abstract too_abstract)
declare -A ABS=(
  [too_geometric]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 0"
  [middle_geometric]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 0.02"
  [normal]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 0.1"
  [middle_abstract]="--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 1.0"
  [too_abstract]="--clip_conv_layer_weights 0 0 0.1 0.1 0 --clip_fc_loss_weight 1.0"
)
mkdir -p jobs/ablation
out=jobs/ablation/all.jsonl; rm -f "$out"
for level in "${LEVELS[@]}"; do
  for n in "${LINES[@]}"; do
    for obj in "${OBJECTS[@]}"; do
      # shellcheck disable=SC2086
      python tools/sweep.py make --datasets "data/objaverse/$obj" --run_name "abl_L${n}_${level}" \
        --runs_root runs --out "$out" --append -- "${COMMON[@]}" --n_curves "$n" ${ABS[$level]} >/dev/null
    done
  done
done
echo "$out: $(wc -l < "$out") jobs (${#OBJECTS[@]} objects x ${#LINES[@]} line counts x ${#LEVELS[@]} levels)"
