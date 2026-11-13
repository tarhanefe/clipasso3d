"""
Export runs for the interactive website grids (CLIPasso3DWebsite/static/data).

    # one experiment family: run dirs <runs_root>/<pattern>/objaverse/<object>
    python tools/export_web.py runs --out ../CLIPasso3DWebsite/static/data --family ablation \\
        --pattern "abl_L{col}_{row}" --rows too_geometric middle_geometric normal middle_abstract too_abstract \\
        --cols 5 10 20 50 100 200
    # shared index (objects, credits, grid axes) and metric tables
    python tools/export_web.py index --out ../CLIPasso3DWebsite/static/data --manifest meshes/objaverse/manifest.json
    python tools/export_web.py metrics --out ../CLIPasso3DWebsite/static/data --pattern "abl_L{col}_{row}" ...
    # connectivity grid: the "off" row reuses the main ablation's normal-level runs
    python tools/export_web.py runs --out ../CLIPasso3DWebsite/static/data --family connect \
        --pattern "conn_L{col}_{row}" --row_pattern "off=abl_L{col}_normal" --rows off weld loss01 loss1 --cols 5 20 50 200

Per object and family it writes <out>/<family>/<object>/:
    runs.json    {"scale": s, "runs": {"<row>|<col>": {curves, snapshots: [batches], final_offset, history_offset}}}
    finals.bin   int16 control points of the final curves of every run (value = int16 * scale / 32767)
    history.bin  int16 control points of up to --max_snapshots snapshots per run (curves x 4 x 3 each)
"""
import argparse
import glob
import json
import os

import numpy as np

LEVEL_LABELS = {"too_geometric": "too geometric", "middle_geometric": "middle geometric", "normal": "normal",
                "middle_abstract": "middle abstract", "too_abstract": "too abstract"}


def label(key):
    return key.split("_(")[0].replace("_", " ")


def read_history(run_dir, max_snapshots):
    """(batches (T,), control points (T,C,4,3)) of a run, snapshots with the final curve count only."""
    h = np.load(os.path.join(run_dir, "curves_history.npz"))
    off, P, batches = h["offsets"], h["P"].astype(np.float32), h["batches"]
    counts = np.diff(off)
    C = counts[-1]
    keep = [i for i in range(len(batches)) if counts[i] == C]
    if len(keep) > max_snapshots:   # evenly spaced, always keeping the first and the last
        sel = np.unique(np.round(np.linspace(0, len(keep) - 1, max_snapshots)).astype(int))
        keep = [keep[i] for i in sel]
    snaps = np.stack([P[off[i]:off[i + 1]] for i in keep])
    return batches[keep].astype(int).tolist(), snaps


def export_family(args):
    objects = sorted({os.path.basename(p) for p in glob.glob(os.path.join(args.runs_root, "*", "objaverse", "*"))})
    if args.objects:
        objects = [o for o in objects if o in args.objects]
    for obj in objects:
        runs, finals, hist = {}, [], []
        for row in args.rows:
            for col in args.cols:
                pattern = dict(rp.split("=", 1) for rp in args.row_pattern).get(row, args.pattern)
                rd = os.path.join(args.runs_root, pattern.format(row=row, col=col), "objaverse", obj)
                if not (os.path.exists(os.path.join(rd, "DONE")) and os.path.exists(os.path.join(rd, "curves_history.npz"))):
                    continue
                batches, snaps = read_history(rd, args.max_snapshots)
                runs[f"{row}|{col}"] = {"curves": int(snaps.shape[1]), "snapshots": batches}
                finals.append(snaps[-1])
                hist.append(snaps)
        if not runs:
            continue
        scale = float(max(np.abs(h).max() for h in hist)) * 1.01
        q = lambda a: np.clip(np.round(a / scale * 32767), -32767, 32767).astype(np.int16)
        f_off = h_off = 0
        for (key, meta), f, h in zip(runs.items(), finals, hist):
            meta["final_offset"], meta["history_offset"] = f_off, h_off
            f_off += f.size
            h_off += h.size
        d = os.path.join(args.out, args.family, obj)
        os.makedirs(d, exist_ok=True)
        np.concatenate([q(f).ravel() for f in finals]).tofile(os.path.join(d, "finals.bin"))
        np.concatenate([q(h).ravel() for h in hist]).tofile(os.path.join(d, "history.bin"))
        with open(os.path.join(d, "runs.json"), "w") as fh:
            json.dump({"object": obj, "scale": scale, "runs": runs}, fh)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6
        print(f"{args.family}/{obj}: {len(runs)} runs, {size:.2f} MB")


def export_index(args):
    manifest = json.load(open(args.manifest)) if args.manifest and os.path.exists(args.manifest) else {}
    objects = sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(args.data_root, "*/"))
                     if os.path.exists(os.path.join(p, "transforms_test.json")))
    index = {
        "objects": [],
        "levels": [{"key": k, "label": LEVEL_LABELS.get(k, label(k))} for k in args.rows],
        "lines": [{"key": str(c), "label": f"{c} lines"} for c in args.cols],
    }
    for o in objects:
        e = {"key": o, "label": label(o)}
        pick = manifest.get(o, {}).get("pick")
        if pick:
            e.update({k: pick[k] for k in ("name", "author", "license", "source")})
        index["objects"].append(e)
    if args.connect_rows:
        index["connect"] = {
            "rows": [{"key": k.split(":")[0], "label": k.split(":", 1)[1] if ":" in k else label(k)} for k in args.connect_rows],
            "cols": [{"key": str(c), "label": f"{c} lines"} for c in args.connect_cols],
        }
    if args.banner_object:
        index["banner"] = {"object": args.banner_object, "lines": args.banner_lines}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    print(f"index.json: {len(objects)} objects")


def export_metrics(args):
    names = [("clip_similarity", "CLIP similarity", True), ("silhouette_precision", "ink on the object", True),
             ("chamfer", "Chamfer distance", False)]
    out = {"rows": [{"key": r, "label": LEVEL_LABELS.get(r, label(r))} for r in args.rows],
           "cols": [{"key": str(c), "label": str(c)} for c in args.cols], "metrics": []}
    cells = {}
    for row in args.rows:
        for col in args.cols:
            ms = [json.load(open(p)) for p in
                  glob.glob(os.path.join(args.runs_root, args.pattern.format(row=row, col=col), "objaverse", "*", "metrics.json"))]
            cells[f"{row}|{col}"] = ms
    for key, lab, hib in names:
        vals = {k: (float(np.mean([m[key] for m in ms if key in m])) if any(key in m for m in ms) else None)
                for k, ms in cells.items()}
        out["metrics"].append({"key": key, "label": lab, "higher_is_better": hib, "values": vals,
                               "n_objects": {k: len(ms) for k, ms in cells.items()}})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"metrics.json: {sum(len(v) for v in cells.values())} runs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", help="runs root folder to export a family, or 'index' / 'metrics'")
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="ablation")
    p.add_argument("--pattern", default="abl_L{col}_{row}", help="Run name pattern with {row} and {col}")
    p.add_argument("--rows", nargs="+", default=list(LEVEL_LABELS))
    p.add_argument("--cols", nargs="+", default=["5", "10", "20", "50", "100", "200"])
    p.add_argument("--objects", nargs="*", default=None)
    p.add_argument("--max_snapshots", type=int, default=40)
    p.add_argument("--row_pattern", nargs="*", default=[],
                   help="row=pattern overrides, e.g. off=abl_L{col}_normal for rows taken from another experiment")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--data_root", default="data/objaverse")
    p.add_argument("--manifest", default="meshes/objaverse/manifest.json")
    p.add_argument("--connect_rows", nargs="*", default=[], help="key:label entries for the connectivity grid")
    p.add_argument("--connect_cols", nargs="*", default=[])
    p.add_argument("--banner_object", default=None)
    p.add_argument("--banner_lines", nargs="*", type=int, default=[5, 20, 50, 200])
    args = p.parse_args()
    if args.what == "index":
        export_index(args)
    elif args.what == "metrics":
        export_metrics(args)
    else:
        args.runs_root = args.what
        export_family(args)


if __name__ == "__main__":
    main()
