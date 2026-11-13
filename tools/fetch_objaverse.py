"""
Pick and download simple, permissively licensed objects from Objaverse (LVIS subset).

For each category the candidates are filtered (license, polygon count, no animation)
and ranked by popularity on Sketchfab as a rough quality signal. The top pick per
category is downloaded to <out>/<category>.glb, and every pick plus a few backups are
recorded in <out>/manifest.json together with the attribution that CC-BY requires.

    pip install objaverse
    python tools/fetch_objaverse.py --out meshes/objaverse
    python tools/fetch_objaverse.py --out meshes/objaverse --swap car_(automobile)   # take the next backup
"""
import argparse
import json
import os
import shutil

import objaverse

# Simple, recognisable shapes that a few dozen strokes can describe
DEFAULT_CATEGORIES = [
    "car_(automobile)", "race_car", "pickup_truck", "school_bus", "jeep",
    "fire_engine", "tractor_(farm_equipment)", "motorcycle", "bicycle", "airplane",
    "helicopter", "boat", "chair", "armchair", "table",
    "bench", "lamp", "guitar", "teapot", "mug",
]
LICENSES = {"cc0", "by"}      # public domain and CC-BY (attribution in manifest.json)


def candidates(uids, ann, min_faces, max_faces):
    out = []
    for uid in uids:
        a = ann.get(uid)
        if a is None or a.get("license") not in LICENSES:
            continue
        if a.get("isAgeRestricted") or a.get("animationCount", 0) > 0:
            continue
        faces = a.get("faceCount") or 0
        if not (min_faces <= faces <= max_faces):
            continue
        out.append(a)
    out.sort(key=lambda a: (a.get("likeCount", 0), a.get("viewCount", 0)), reverse=True)
    return out


def record(a, category):
    user = a.get("user") or {}
    return {
        "category": category,
        "uid": a["uid"],
        "name": a.get("name"),
        "license": a.get("license"),
        "author": user.get("displayName") or user.get("username"),
        "source": a.get("viewerUrl") or f"https://sketchfab.com/3d-models/{a['uid']}",
        "faces": a.get("faceCount"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="meshes/objaverse")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p.add_argument("--min_faces", type=int, default=1000)
    p.add_argument("--max_faces", type=int, default=60000)
    p.add_argument("--backups", type=int, default=3, help="Backup candidates kept per category")
    p.add_argument("--swap", nargs="*", default=[], help="Categories whose current pick is replaced by the next backup")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    man_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}

    lvis = objaverse.load_lvis_annotations()
    todo = [c for c in args.categories if c not in manifest or c in args.swap]
    if todo:
        uids = sorted({u for c in todo for u in lvis.get(c, [])})
        ann = objaverse.load_annotations(uids)
        for c in todo:
            rejected = set(manifest.get(c, {}).get("rejected", []))
            if c in args.swap and c in manifest:
                rejected.add(manifest[c]["pick"]["uid"])
            cands = [a for a in candidates(lvis.get(c, []), ann, args.min_faces, args.max_faces)
                     if a["uid"] not in rejected]
            if not cands:
                print(f"[fetch] {c}: no candidate passes the filters, skipped")
                continue
            manifest[c] = {
                "pick": record(cands[0], c),
                "backups": [record(a, c) for a in cands[1:1 + args.backups]],
                "rejected": sorted(rejected),
            }

    picks = {c: m["pick"] for c, m in manifest.items() if c in args.categories}
    paths = objaverse.load_objects(uids=[m["uid"] for m in picks.values()])
    for c, m in picks.items():
        dst = os.path.join(args.out, f"{c}.glb")
        if c in args.swap or not os.path.exists(dst):
            shutil.copy(paths[m["uid"]], dst)
        print(f"[fetch] {c:28s} {m['faces']:>6} faces  {m['license']:4s}  {m['name']}  ({m['author']})")

    tmp = man_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, man_path)
    print(f"[fetch] {len(picks)} objects in {args.out}")


if __name__ == "__main__":
    main()
