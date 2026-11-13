"""
Progress of training sweeps (tools/sweep.py): jobs done / total, running workers, failures,
ETA and the best results so far.

    python tools/status.py jobs/baseline.jsonl [more.jsonl ...] [--stale 600]
"""
import argparse
import json
import os
import time


def last_json_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            lines = [l for l in f.read().decode(errors="ignore").splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except (OSError, ValueError):
        return None


def batch_times(path, n=50):
    """Seconds per batch from the last `n` log lines (continuous segment only)."""
    try:
        with open(path) as f:
            rows = [json.loads(l) for l in f.readlines()[-n:] if l.strip()]
    except (OSError, ValueError):
        return []
    dts = [b["time"] - a["time"] for a, b in zip(rows, rows[1:]) if b["batch"] == a["batch"] + 1]
    return [d for d in dts if d > 0]


def fmt_time(s):
    if s is None:
        return "n/a"
    if s < 60:
        return f"{int(s)}s"
    h, m = divmod(int(s) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs", nargs="+")
    p.add_argument("--stale", type=float, default=600)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    jobs = []
    for jf in args.jobs:
        with open(jf) as f:
            jobs += [json.loads(l) for l in f if l.strip()]

    counts = {"done": 0, "running": 0, "stale": 0, "failed": 0, "pending": 0}
    remaining_batches, dts, running_rows, failures, results = 0, [], [], [], []
    now = time.time()
    for j in jobs:
        rd = j["run_dir"]
        last = last_json_line(os.path.join(rd, "log.jsonl"))
        done_b, total_b = (last["batch"], last["total"]) if last else (0, None)
        if os.path.exists(os.path.join(rd, "DONE")):
            counts["done"] += 1
            try:
                results.append((rd, json.load(open(os.path.join(rd, "metrics.json")))))
            except (OSError, ValueError):
                pass
            continue
        if os.path.exists(os.path.join(rd, "FAILED")):
            counts["failed"] += 1
            msg = open(os.path.join(rd, "FAILED"), errors="ignore").read().strip().splitlines()
            failures.append((rd, msg[-1] if msg else "?"))
            continue
        if total_b is not None:
            remaining_batches += total_b - done_b
        claim = os.path.join(rd, ".claim")
        if os.path.isdir(claim):
            hb = os.path.join(claim, "heartbeat")
            age = now - os.path.getmtime(hb if os.path.exists(hb) else claim)
            state = "running" if age < args.stale else "stale"
            counts[state] += 1
            owner = open(os.path.join(claim, "owner")).read().strip() if os.path.exists(os.path.join(claim, "owner")) else "?"
            running_rows.append(f"  {state:7s} {rd}  {done_b}/{total_b or '?'}  {owner}")
            if state == "running":
                dts += batch_times(os.path.join(rd, "log.jsonl"))
        else:
            counts["pending"] += 1
        n_fail = os.path.join(rd, "failures.txt")
        if os.path.exists(n_fail):
            failures.append((rd, f"{len(open(n_fail).readlines())} failed attempt(s), will retry"))

    total = len(jobs)
    print(f"jobs: {counts['done']}/{total} done | running {counts['running']} | stale {counts['stale']} "
          f"| pending {counts['pending']} | failed {counts['failed']}")
    if running_rows:
        print("\n".join(running_rows))

    # ETA: mean seconds per batch of running jobs; pending jobs assumed as long as the average known job
    eta = None
    if dts and counts["running"]:
        sec_per_batch = sum(dts) / len(dts)
        known = [last_json_line(os.path.join(j["run_dir"], "log.jsonl")) for j in jobs]
        totals = [k["total"] for k in known if k]
        pending_batches = counts["pending"] * (sum(totals) / len(totals) if totals else 0)
        eta = (remaining_batches + pending_batches) * sec_per_batch / counts["running"]
        print(f"speed: {sec_per_batch:.1f} s/batch per worker   ETA: {fmt_time(eta)}")
    else:
        print("ETA: n/a (no running worker with recent progress)")

    if failures:
        print("failures:")
        for rd, msg in failures[:20]:
            print(f"  {rd}: {msg}")

    if results:
        keys = ["clip_similarity", "silhouette_precision", "chamfer"]
        print(f"best results (by clip_similarity, top {args.top}):")
        results.sort(key=lambda r: r[1].get("clip_similarity", 0), reverse=True)
        for rd, m in results[:args.top]:
            vals = "  ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)
            print(f"  {rd}  {vals}")
        means = {k: sum(m[k] for _, m in results if k in m) / max(1, sum(k in m for _, m in results)) for k in keys}
        print("  mean over finished: " + "  ".join(f"{k}={v:.3f}" for k, v in means.items()))


if __name__ == "__main__":
    main()
