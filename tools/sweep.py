"""
Run a list of training jobs with any number of workers that share the work through the
file system (several pods / GPUs, all restart-safe).

Make a job list (one run per dataset, extra args are passed to run.py):
    python tools/sweep.py make --datasets data/objaverse/* --run_name baseline \\
        --out jobs/baseline.jsonl -- --epochs 10 --n_curves 25

Run a worker (worker W of N; usually started by the pod entry script, one per GPU):
    python tools/sweep.py run --jobs jobs/baseline.jsonl --worker 0 --nworkers 4

Each job trains with `python run.py ... --out_dir <run_dir>`, which resumes from its own
checkpoint. A worker claims a job with an atomic mkdir of <run_dir>/.claim and refreshes a
heartbeat while it runs; claims whose heartbeat is older than --stale seconds belong to a dead
worker and are taken over. A job that fails --max_failures times is marked FAILED and skipped.
CUDA / driver errors stop the worker with a non-zero exit (its CUDA context may be broken),
so the restart loop starts a fresh process and the job is redone. Progress: tools/status.py.
"""
import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFRA_ERRORS = ("CUDA out of memory", "CUDA error", "device-side assert", "cuDNN error",
                "NCCL error", "Expected all tensors to be on the same device", "driver")


def make_jobs(args, extra):
    jobs = []
    for d in sorted(sum((glob.glob(p) for p in args.datasets), [])):
        if not os.path.exists(os.path.join(d, "transforms_train.json")):
            continue
        rel = os.path.relpath(d, args.data_root)
        run_dir = os.path.join(args.runs_root, args.run_name, rel)
        jobs.append({"run_dir": run_dir,
                     "args": ["--data_root", args.data_root, "--data_name", rel, "--run_name", args.run_name] + extra})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.append and os.path.exists(args.out):
        done = {j["run_dir"] for j in load_jobs(args.out)}
        jobs = load_jobs(args.out) + [j for j in jobs if j["run_dir"] not in done]
    with open(args.out + ".tmp", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    os.replace(args.out + ".tmp", args.out)
    print(f"{len(jobs)} jobs -> {args.out}")


def load_jobs(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def heartbeat_age(claim):
    try:
        return time.time() - os.path.getmtime(os.path.join(claim, "heartbeat"))
    except OSError:
        try:                                    # claimed but no heartbeat yet
            return time.time() - os.path.getmtime(claim)
        except OSError:
            return None


def try_claim(run_dir, stale, me):
    os.makedirs(run_dir, exist_ok=True)
    claim = os.path.join(run_dir, ".claim")
    try:
        os.mkdir(claim)
    except FileExistsError:
        age = heartbeat_age(claim)
        if age is None or age < stale:
            return None
        # dead worker: take over by renaming its claim away (atomic, only one taker wins)
        dead = f"{claim}.dead.{int(time.time())}.{os.getpid()}"
        try:
            os.rename(claim, dead)
        except OSError:
            return None
        shutil.rmtree(dead, ignore_errors=True)
        try:
            os.mkdir(claim)
        except FileExistsError:
            return None
    with open(os.path.join(claim, "owner"), "w") as f:
        f.write(me)
    open(os.path.join(claim, "heartbeat"), "w").close()
    return claim


def release(claim):
    shutil.rmtree(claim, ignore_errors=True)


def run_job(job, claim, log_path):
    stop = threading.Event()

    def beat():
        while not stop.wait(30):
            try:
                os.utime(os.path.join(claim, "heartbeat"))
            except OSError:
                pass

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    cmd = [sys.executable, "-u", os.path.join(ROOT, "run.py"), *job["args"], "--out_dir", job["run_dir"]]
    with open(log_path, "a") as log:
        log.write(f"\n==== {time.ctime()} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    stop.set()
    t.join()
    return proc.returncode


def tail(path, n=60):
    try:
        with open(path, errors="ignore") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def run_worker(args):
    jobs = load_jobs(args.jobs)
    me = f"{socket.gethostname()}:{os.getpid()}:w{args.worker}"
    # each worker starts at a different offset to avoid contention on the same claims
    order = jobs[args.worker::args.nworkers] + [j for i, j in enumerate(jobs) if i % args.nworkers != args.worker]
    while True:
        claimed_any = False
        for job in order:
            rd = job["run_dir"]
            if os.path.exists(os.path.join(rd, "DONE")) or os.path.exists(os.path.join(rd, "FAILED")):
                continue
            claim = try_claim(rd, args.stale, me)
            if claim is None:
                continue
            claimed_any = True
            log_path = os.path.join(rd, "train.log")
            print(f"[{me}] start {rd}", flush=True)
            t0 = time.time()
            rc = run_job(job, claim, log_path)
            if rc == 0:
                print(f"[{me}] done  {rd} in {(time.time() - t0) / 60:.1f} min", flush=True)
                release(claim)
                continue
            out = tail(log_path)
            if any(e in out for e in INFRA_ERRORS):
                # not the job's fault: leave it for a fresh process, stop this one
                print(f"[{me}] infrastructure error in {rd}, exiting for a clean restart:\n{out[-2000:]}", flush=True)
                release(claim)
                sys.exit(3)
            fails_path = os.path.join(rd, "failures.txt")
            with open(fails_path, "a") as f:
                f.write(f"{time.ctime()} rc={rc} {me}\n")
            n_fail = len(open(fails_path).readlines())
            print(f"[{me}] FAIL  {rd} (rc={rc}, failure {n_fail}/{args.max_failures})", flush=True)
            if n_fail >= args.max_failures:
                with open(os.path.join(rd, "FAILED"), "w") as f:
                    f.write(out[-5000:])
            release(claim)
        if claimed_any:
            continue
        # nothing claimable right now: only stop when every job is finished. Jobs claimed by other
        # workers may still become free (a preempted pod leaves claims behind that go stale after
        # --stale seconds), so keep checking until then.
        pending = [j["run_dir"] for j in jobs
                   if not any(os.path.exists(os.path.join(j["run_dir"], m)) for m in ("DONE", "FAILED"))]
        if not pending:
            break
        print(f"[{me}] {len(pending)} jobs claimed by other workers, checking again in {args.poll:.0f} s", flush=True)
        time.sleep(args.poll)
    print(f"[{me}] all jobs finished", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="Write a job list (args after -- go to run.py)")
    m.add_argument("--datasets", nargs="+", required=True, help="Dataset folders or globs")
    m.add_argument("--data_root", default="data")
    m.add_argument("--runs_root", default="runs")
    m.add_argument("--run_name", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--append", action="store_true", help="Add to an existing job list (same run_dir is not duplicated)")
    r = sub.add_parser("run", help="Run a worker")
    r.add_argument("--jobs", required=True)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--nworkers", type=int, default=1)
    r.add_argument("--stale", type=float, default=600, help="Seconds without heartbeat before a claim is dead")
    r.add_argument("--max_failures", type=int, default=3)
    r.add_argument("--poll", type=float, default=60, help="Seconds between checks while other workers hold all jobs")

    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)
    if args.cmd == "make":
        make_jobs(args, extra)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
