#!/bin/bash
# Render every mesh in MESH_DIR into OUT_DIR/<mesh name>/ with tools/render_mesh.py.
# Resumable: finished datasets (all three transforms_*.json present) are skipped, and
# render_mesh.py skips views that already exist, so after a crash just run it again.
#
#   tools/render_all.sh meshes/objaverse data/objaverse [extra render_mesh.py args]
#
# Blender: uses $BLENDER if set, else `blender` on PATH, else the macOS app bundle.
# Progress: python tools/render_status.py data/objaverse
set -uo pipefail
MESH_DIR=${1:?mesh folder}
OUT_DIR=${2:?output folder}
shift 2
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -z "${BLENDER:-}" ]; then
  if command -v blender >/dev/null; then BLENDER=blender
  elif [ -x /Applications/Blender.app/Contents/MacOS/Blender ]; then BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
  else echo "Blender not found: set BLENDER=/path/to/blender"; exit 2; fi
fi

mkdir -p "$OUT_DIR/logs"
fail=0
shopt -s nullglob nocaseglob
for mesh in "$MESH_DIR"/*.{glb,gltf,obj,fbx,ply,stl}; do
  name=$(basename "${mesh%.*}")
  out="$OUT_DIR/$name"
  if [ -f "$out/transforms_train.json" ] && [ -f "$out/transforms_val.json" ] && [ -f "$out/transforms_test.json" ]; then
    echo "[render_all] $name: done, skipping"; continue
  fi
  echo "[render_all] $name: rendering"
  if ! "$BLENDER" -b --factory-startup -P "$HERE/render_mesh.py" -- --mesh "$mesh" --out "$out" "$@" \
        > "$OUT_DIR/logs/$name.log" 2>&1; then
    echo "[render_all] $name: FAILED (see $OUT_DIR/logs/$name.log)"; fail=1
  fi
done
exit $fail
