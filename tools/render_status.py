"""
Progress of tools/render_all.sh: views rendered / expected per object, failures and an ETA.

    python tools/render_status.py data/objaverse [--per_object 160]
"""
import argparse
import glob
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--per_object", type=int, default=160, help="Views per object (n_train + n_val + n_test)")
    p.add_argument("--mesh_dir", default="meshes/objaverse", help="Used to count objects not started yet")
    args = p.parse_args()

    names = set()
    if os.path.isdir(args.mesh_dir):
        names |= {os.path.splitext(f)[0] for f in os.listdir(args.mesh_dir)
                  if f.lower().endswith((".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl"))}
    names |= {d for d in os.listdir(args.out_dir) if os.path.isdir(os.path.join(args.out_dir, d)) and d != "logs"}

    done_views, mtimes, rows = 0, [], []
    for n in sorted(names):
        d = os.path.join(args.out_dir, n)
        pngs = glob.glob(os.path.join(d, "rgb", "*", "r_*.png"))
        mtimes += [os.path.getmtime(f) for f in pngs]
        finished = all(os.path.exists(os.path.join(d, f"transforms_{s}.json")) for s in ("train", "val", "test"))
        log = os.path.join(args.out_dir, "logs", n + ".log")
        failed = os.path.exists(log) and "Traceback" in open(log, errors="ignore").read()
        n_views = args.per_object if finished else len(pngs)
        done_views += n_views
        state = "done" if finished else ("FAILED" if failed else ("running" if pngs else "pending"))
        rows.append(f"  {n:30s} {n_views:4d}/{args.per_object}  {state}")

    total = len(names) * args.per_object
    n_done = sum(r.endswith("done") for r in rows)
    print(f"objects: {n_done}/{len(names)} done   views: {done_views}/{total}")
    print("\n".join(rows))
    recent = sorted(mtimes)[-200:]
    if len(recent) > 10 and time.time() - recent[-1] < 600:
        rate = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)   # views per second, recent window
        eta = (total - done_views) / rate
        print(f"rate: {rate * 60:.1f} views/min   ETA: {eta / 60:.1f} min")
    else:
        print("ETA: n/a (no views written in the last 10 min)")


if __name__ == "__main__":
    main()
