#!/usr/bin/env python3
"""
GPU job queue: run many independent experiments concurrently across GPUs.

The EXACT models are tiny (< 2 MB), so one run cannot use a whole 4090.
Instead of DDP for a single run, this runner keeps N independent runs
in flight per GPU, each pinned with CUDA_VISIBLE_DEVICES.

Usage:
    python experiments/runner.py experiments/jobs/table2_loocv.json \
        --per_gpu 3 --gpus 0,1,2,3

    # Dry run: print what would run and what is already done
    python experiments/runner.py experiments/jobs/table2_loocv.json --dry_run

Job file format (JSON list):
    [{"name": "unet-physiq-42",
      "cmd":  "python main_meta_v2.py --model unet --dataset physiq --seed 42",
      "done_if": "results/runs/unet-physiq-42.json",   # optional: skip if exists
      "after": ["other-job-name"]}]                    # optional: dependencies

Each job's stdout/stderr goes to logs/<name>.log; a status table is kept in
logs/_status.json and reprinted every time a job finishes, so `tail -f` or a
glance at the file is enough to see where a long sweep is.
"""
import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime


def detect_gpus():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return ["0"]


def load_jobs(path):
    with open(path) as f:
        jobs = json.load(f)
    if isinstance(jobs, dict) and "jobs" in jobs:
        jobs = jobs["jobs"]
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)), "job names must be unique"
    return jobs


class Runner:
    def __init__(self, jobs, gpus, per_gpu, log_dir, retries, dry_run, cwd):
        self.jobs = {j["name"]: j for j in jobs}
        self.order = [j["name"] for j in jobs]
        self.gpus = gpus
        self.per_gpu = per_gpu
        self.log_dir = log_dir
        self.retries = retries
        self.dry_run = dry_run
        self.cwd = cwd
        self.status = {}
        self.lock = threading.Lock()
        self.q = queue.Queue()
        os.makedirs(log_dir, exist_ok=True)

        for name in self.order:
            j = self.jobs[name]
            done_if = j.get("done_if")
            if done_if and os.path.exists(os.path.join(cwd, done_if)):
                self.status[name] = {"state": "done", "note": "done_if exists"}
            else:
                self.status[name] = {"state": "pending"}
        self._write_status()

    # ---------------------------------------------------------------- helpers
    def _write_status(self):
        with open(os.path.join(self.log_dir, "_status.json"), "w") as f:
            json.dump(self.status, f, indent=2)

    def _deps_ok(self, name):
        for d in self.jobs[name].get("after", []):
            if self.status.get(d, {}).get("state") != "done":
                return False
        return True

    def _deps_failed(self, name):
        return any(
            self.status.get(d, {}).get("state") == "failed"
            for d in self.jobs[name].get("after", [])
        )

    def summary(self):
        counts = {}
        for s in self.status.values():
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        return " ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    # ------------------------------------------------------------------ worker
    def _run_one(self, name, gpu):
        j = self.jobs[name]
        cmd = j["cmd"]
        if isinstance(cmd, list):
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
        else:
            cmd_str = cmd
        log_path = os.path.join(self.log_dir, f"{name}.log")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("WANDB_SILENT", "true")
        env.setdefault("PYTHONUNBUFFERED", "1")
        # each run is launch-bound; without this every process spawns 32 BLAS
        # threads and they thrash each other
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.update(j.get("env", {}))

        attempts = 0
        while True:
            attempts += 1
            t0 = time.time()
            with self.lock:
                self.status[name] = {
                    "state": "running", "gpu": gpu, "attempt": attempts,
                    "started": datetime.now().isoformat(timespec="seconds"),
                }
                self._write_status()
            with open(log_path, "a") as lf:
                lf.write(f"\n===== [{datetime.now()}] attempt {attempts} on GPU {gpu}\n{cmd_str}\n\n")
                lf.flush()
                rc = subprocess.call(
                    cmd_str, shell=True, cwd=self.cwd, env=env,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
            elapsed = time.time() - t0
            done_if = j.get("done_if")
            ok = rc == 0 and (not done_if or os.path.exists(os.path.join(self.cwd, done_if)))
            if ok:
                state, note = "done", f"{elapsed/60:.1f} min"
            elif attempts <= self.retries:
                print(f"[runner] {name} failed (rc={rc}), retrying", flush=True)
                continue
            else:
                state, note = "failed", f"rc={rc} after {attempts} attempts; see {log_path}"
            with self.lock:
                self.status[name] = {"state": state, "gpu": gpu, "note": note,
                                     "elapsed_sec": round(elapsed, 1)}
                self._write_status()
            print(f"[runner] {name}: {state} ({note})  |  {self.summary()}", flush=True)
            return

    def _worker(self, gpu, slot):
        while True:
            try:
                name = self.q.get(timeout=1)
            except queue.Empty:
                if self._all_settled():
                    return
                continue
            if name is None:
                return
            # optional dynamic skip list: one job name (or glob) per line in logs/_skip.txt,
            # consulted right before launch so the queue can be trimmed without a restart
            if self._skipped(name):
                with self.lock:
                    self.status[name] = {"state": "skipped", "note": "listed in _skip.txt"}
                    self._write_status()
                self.q.task_done()
                continue
            # wait for dependencies
            while not self._deps_ok(name):
                if self._deps_failed(name):
                    with self.lock:
                        self.status[name] = {"state": "failed", "note": "dependency failed"}
                        self._write_status()
                    break
                time.sleep(5)
            else:
                self._run_one(name, gpu)
            self.q.task_done()

    def _skipped(self, name):
        import fnmatch
        path = os.path.join(self.log_dir, "_skip.txt")
        if not os.path.exists(path):
            return False
        pats = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
        return any(fnmatch.fnmatch(name, p) for p in pats)

    def _all_settled(self):
        return all(s["state"] in ("done", "failed", "skipped") for s in self.status.values())

    # -------------------------------------------------------------------- main
    def run(self):
        pending = [n for n in self.order if self.status[n]["state"] == "pending"]
        print(f"[runner] {len(pending)} pending, {len(self.order) - len(pending)} already done; "
              f"GPUs={self.gpus} per_gpu={self.per_gpu}", flush=True)
        if self.dry_run:
            for n in pending:
                c = self.jobs[n]["cmd"]
                print(f"  {n}: {c if isinstance(c, str) else ' '.join(c)}")
            return 0
        for n in pending:
            self.q.put(n)
        threads = []
        for gpu in self.gpus:
            for slot in range(self.per_gpu):
                t = threading.Thread(target=self._worker, args=(gpu, slot), daemon=True)
                t.start()
                threads.append(t)
                time.sleep(2)  # stagger start-up so dataset loading does not thrash
        for t in threads:
            t.join()
        failed = [n for n, s in self.status.items() if s["state"] == "failed"]
        print(f"[runner] finished: {self.summary()}", flush=True)
        if failed:
            print("[runner] FAILED jobs:\n  " + "\n  ".join(failed), flush=True)
        return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", help="JSON job file (see experiments/make_jobs.py)")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--per_gpu", type=int, default=4, help="concurrent jobs per GPU")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--cwd", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="repo root to run commands from")
    a = ap.parse_args()
    gpus = a.gpus.split(",") if a.gpus else detect_gpus()
    jobs = load_jobs(a.jobs)
    r = Runner(jobs, gpus, a.per_gpu, os.path.join(a.cwd, a.log_dir), a.retries, a.dry_run, a.cwd)
    sys.exit(r.run())


if __name__ == "__main__":
    main()
