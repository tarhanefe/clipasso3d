#!/bin/bash
# Connectivity ablation: can the sketches be built as one connected wire sculpture?
# Normal abstraction (CLIPasso balance), lines 5/20/50/200, every rendered object.
#   (no constraint)  = runs/abl_L<n>_normal from experiments/ablation.sh (not rerun)
#   weld             = those final curves, welded into one piece after training (no training)
#   loss01 / loss1   = connectivity loss with weight 0.1 / 1.0 from batch 490 on, then welded
# The weld translates separate clusters rigidly until they touch (contact distance 2 x thickness);
# metrics.json reports pieces before (pre_weld_*) and after the weld and the total weld shift.
# Usage (bash >= 4, on the pods, after the main ablation): experiments/connectivity.sh [objects...]
set -euo pipefail
cd "$(dirname "$0")/.."
OBJECTS=("$@")
if [ ${#OBJECTS[@]} -eq 0 ]; then
  for d in data/objaverse/*/; do [ -f "$d/transforms_test.json" ] && OBJECTS+=("$(basename "$d")"); done
fi
WANDB=(--wandb --wandb_mode offline --wandb_entity efe-tarhan-epfl --wandb_project clipasso3d)
STOP=(--epochs 600 --eval_interval 100 --early_stop_patience 5 --early_stop_tol 0.005 --min_batches 500)
PRUNE=(--prune_mode respawn --prune_start 490 --prune_every 195 --prune_stop 2450)
NORMAL=(--clip_conv_layer_weights 0 0 1 1 0 --clip_fc_loss_weight 0.1)
COMMON=(--thickness 0.008 --snapshot_every 20 --frame_every 10 --gif_fps 30 --rotation_time 6 "${NORMAL[@]}" "${WANDB[@]}")
LINES=(5 20 50 200)
mkdir -p jobs/connect
out=jobs/connect/all.jsonl; rm -f "$out"
for n in "${LINES[@]}"; do
  for obj in "${OBJECTS[@]}"; do
    ds="data/objaverse/$obj"
    python tools/sweep.py make --datasets "$ds" --run_name "conn_L${n}_weld" --runs_root runs --out "$out" --append -- \
      "${COMMON[@]}" --n_curves "$n" --epochs 0 --weld --init_from "runs/abl_L${n}_normal/{data_name}/curves.pt" >/dev/null
    for w in 01:0.1 1:1.0; do
      python tools/sweep.py make --datasets "$ds" --run_name "conn_L${n}_loss${w%%:*}" --runs_root runs --out "$out" --append -- \
        "${COMMON[@]}" "${STOP[@]}" "${PRUNE[@]}" --n_curves "$n" --connect_weight "${w##*:}" --connect_start 490 >/dev/null
    done
  done
done
echo "$out: $(wc -l < "$out") jobs"
