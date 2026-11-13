#!/bin/bash
# Round 1: which reconstruction changes help? One job list per object (one pod per object).
#   A_old     original settings (random sphere init, white on black, length-based sampling, 1 view x 30 steps)
#   B_fixed   A + 64 spheres per curve, batched (speed; should not change quality)
#   C_bw      B + black strokes on white
#   D_hull    B + visual-hull initialization (longer initial strokes)
#   E_mv      B + 4 views per step, 1 step per batch
#   F_all     B + C + D + E
# Usage: experiments/round1.sh [objects...]   (default: 6 diverse objects)
set -euo pipefail
cd "$(dirname "$0")/.."
OBJECTS=("$@"); [ ${#OBJECTS[@]} -gt 0 ] || OBJECTS=("car_(automobile)" airplane chair teapot bicycle guitar)
WANDB=(--wandb --wandb_entity efe-tarhan --wandb_project clipasso3d)
COMMON=(--n_curves 25 --thickness 0.02 --gif_fps 30 --rotation_time 6 "${WANDB[@]}")
OLD=(--epochs 10 --batch_size 1 --inner_steps 30 --frame_every 5)
MV=(--epochs 120 --batch_size 4 --inner_steps 1 --frame_every 15)
FIX=(--samples_per_curve 64)
HULL=(--init hull --length 0.15)
BW=(--style black_on_white)

declare -A CFG=(
  [A_old]="${OLD[*]}"
  [B_fixed]="${OLD[*]} ${FIX[*]}"
  [C_bw]="${OLD[*]} ${FIX[*]} ${BW[*]}"
  [D_hull]="${OLD[*]} ${FIX[*]} ${HULL[*]}"
  [E_mv]="${MV[*]} ${FIX[*]}"
  [F_all]="${MV[*]} ${FIX[*]} ${HULL[*]} ${BW[*]}"
)
mkdir -p jobs/round1
for obj in "${OBJECTS[@]}"; do
  out="jobs/round1/${obj}.jsonl"; rm -f "$out"
  for name in A_old B_fixed C_bw D_hull E_mv F_all; do
    # shellcheck disable=SC2086
    python tools/sweep.py make --datasets "data/objaverse/$obj" --run_name "round1_$name" \
      --runs_root runs --out "$out" --append -- "${COMMON[@]}" ${CFG[$name]} >/dev/null
  done
  echo "$out: $(wc -l < "$out") jobs"
done
