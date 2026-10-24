#!/usr/bin/env python3
"""Campaign runner for the NUG30 Laboratory.

Reads job requests from campagne/file.jsonl (one JSON object per line:
{"id", "instance", "args", "label", "desc"}), runs at most MAXJ jobs at a
time with the C solver, stores each run in runs/<id>/ (cmd, progress,
result, stderr, meta) and rewrites etat/etat.json every 5 seconds for the
live dashboard of the site.
"""
import json, os, subprocess, time, shutil, socket, sys

ROOT = os.environ.get("NUG30_ROOT", "/mnt/project-files/nug30")
BIN = os.environ.get("QAP_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "qap_bb"))
MAXJ = int(os.environ.get("MAXJ", "3"))
QUEUE = f"{ROOT}/campagne/file.jsonl"
ETAT = f"{ROOT}/etat/etat.json"
AGENTS = f"{ROOT}/etat/agents.json"

def rjson(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return default

def wjson(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)

def load_queue():
    jobs = []
    try:
        for line in open(QUEUE):
            line = line.strip()
            if line: jobs.append(json.loads(line))
    except FileNotFoundError: pass
    return jobs

procs = {}  # id -> Popen

def start(job):
    d = f"{ROOT}/runs/{job['id']}"
    os.makedirs(d, exist_ok=True)
    cmd = [BIN] + job.get("mode", "solve").split() + [f"{ROOT}/data/{job['instance']}.dat"] + job.get("args", "").split() \
          + ["-prog", f"{d}/progress.json"]
    with open(f"{d}/cmd.txt", "w") as f: f.write(" ".join(cmd) + "\n")
    out = open(f"{d}/result.json", "w"); err = open(f"{d}/stderr.log", "w")
    p = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
    procs[job["id"]] = p
    wjson(f"{d}/meta.json", {**job, "status": "running", "start": time.time(), "host": socket.gethostname(), "pid": p.pid})

def job_state(job):
    d = f"{ROOT}/runs/{job['id']}"
    meta = rjson(f"{d}/meta.json")
    if meta is None: return "pending", None
    return meta.get("status", "running"), meta

def main():
    cpu = ""
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"): cpu = l.split(":", 1)[1].strip(); break
    except Exception: pass
    while True:
        jobs = load_queue()
        # finished processes
        for jid, p in list(procs.items()):
            if p.poll() is not None:
                d = f"{ROOT}/runs/{jid}"
                meta = rjson(f"{d}/meta.json", {})
                meta["status"] = "done" if p.returncode == 0 else f"failed({p.returncode})"
                meta["end"] = time.time()
                wjson(f"{d}/meta.json", meta)
                del procs[jid]
        running = len(procs)
        for job in jobs:
            st, meta = job_state(job)
            if st == "running" and job["id"] not in procs:
                # orphan from a previous runner: mark as interrupted
                meta["status"] = "interrupted"; wjson(f"{ROOT}/runs/{job['id']}/meta.json", meta)
            if st == "pending" and job.get("hold") is not True and running < MAXJ:
                start(job); running += 1
        # state file for the site
        rows = []
        for job in jobs:
            d = f"{ROOT}/runs/{job['id']}"
            st, meta = job_state(job)
            row = {k: job.get(k) for k in ("id", "instance", "label", "desc", "args", "mode", "group")}
            row["status"] = st
            if meta:
                row["start"] = meta.get("start"); row["end"] = meta.get("end")
            prog = rjson(f"{d}/progress.json")
            if prog: row["progress"] = prog
            res = None
            try:
                txt = open(f"{d}/result.json").read().strip()
                if txt: res = json.loads(txt.splitlines()[-1])
            except Exception: res = None
            if res: row["result"] = res
            rows.append(row)
        try: load = os.getloadavg()
        except Exception: load = None
        wjson(ETAT, {"updated": time.time(), "machine": {"cpu": cpu, "cores": os.cpu_count(), "load": load, "maxj": MAXJ},
                     "jobs": rows, "agents": rjson(AGENTS, [])})
        time.sleep(5)

if __name__ == "__main__":
    main()
