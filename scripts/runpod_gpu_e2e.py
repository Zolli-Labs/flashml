#!/usr/bin/env python3
"""scripts/runpod_gpu_e2e.py — rent ONE RunPod 2-GPU box, prove flashruntime's
CUDA / nccl code paths on real hardware, then always terminate.

WHY: the CPU test suite exercises `flashruntime.torch._resolve_device` and the
nccl branch of `prepare()` only through a *pure* helper — no GPU is present on a
dev laptop. This harness closes that gap end-to-end: it builds the wheel, rents
the cheapest available 2-GPU community box among (RTX 4090, A5000, 3090), runs
`tests/test_gpu_e2e.py` and two GPU benchmark scenarios on it, brings the
results home, and tears the box down.

API SURFACE USED (recorded per Task 12):
  * Offer discovery + pricing — RunPod **GraphQL** (https://api.runpod.io/graphql).
    The REST API (rest.runpod.io/v1, confirmed against its own openapi.json)
    exposes NO gpuTypes endpoint, so the 2-GPU community price + stock is only
    queryable over GraphQL's `gpuTypes { communityPrice, lowestPrice(input){
    uninterruptablePrice, stockStatus } }`.
  * Pod lifecycle (create / get / list / delete) — RunPod **REST v1**
    (rest.runpod.io/v1), the current recommended surface for pod CRUD: create
    returns the machine + `portMappings` we SSH into; delete is DELETE
    /v1/pods/{id} → 204. (The legacy GraphQL podFindAndDeployOnDemand mutation
    still works but REST is what RunPod's current docs steer new code to.)
  Auth on BOTH surfaces is a Bearer token (RUNPOD_API_KEY), read from env/.env,
  sent ONLY in the Authorization header, and never printed/logged/written.

SSH: RunPod's official images run sshd and import a public key from the
`PUBLIC_KEY` env var (we also set `SSH_PUBLIC_KEY`). We generate an EPHEMERAL
ed25519 keypair per run, inject the public half, expose 22/tcp on a public IP,
and connect as root with the private half — nothing touches ~/.ssh.

SAFETY (hard rules, not suggestions):
  * ONE pod at a time — any pre-existing flashruntime-e2e-* pod is swept first.
  * A 30-min hard deadline anchored at pod CREATION (that is when billing
    starts), checked around every phase; each remote step is also given a
    bounded, deadline-clamped subprocess timeout.
  * Abort-and-terminate on ANY failed step (no retry-loop; one reconnect while
    sshd comes up is the only retry).
  * A `finally:` that deletes the pod AND lists+deletes any labeled straggler.
  * Target < $1 total (cheapest tier is ~$0.44/hr for 2×3090 → <$0.25 for a
    30-min cap; real runs finish in minutes).

Dev-only; not shipped in the wheel. Needs `httpx` (the service extra) + ssh/scp.

Usage:
    python scripts/runpod_gpu_e2e.py --plan-only   # query offers, create nothing
    python scripts/runpod_gpu_e2e.py               # the real, money-spending run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "scripts" / "runpod_results"

GQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"

# CUDA >= 12 PyTorch image (verified on Docker Hub): CUDA 12.8.1, torch 2.7.1.
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204"
# Priority order is by community price ascending; the actual pick is the
# cheapest of these that has 2-GPU community STOCK at query time.
TARGET_GPUS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
]
GPU_COUNT = 2
NAME_PREFIX = "flashruntime-e2e-"

DEADLINE_S = 30 * 60  # hard cap on paid pod lifetime
PROVISION_BUDGET_S = 15 * 60  # max wait for RUNNING + public IP + port 22
SSH_WAIT_BUDGET_S = 3 * 60  # max wait for sshd after RUNNING

SSH_COMMON = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
]


# --------------------------------------------------------------------------
# tiny logging + timeline (never contains the API key)
# --------------------------------------------------------------------------
class Runlog:
    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        stamp = f"[{time.monotonic() - self.t0:7.1f}s | {_utcnow()}] {msg}"
        print(stamp, flush=True)
        self.lines.append(stamp)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%SZ")


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def load_api_key() -> str:
    """RUNPOD_API_KEY from the process env, else the gitignored repo .env.
    Refuses to run if absent. The value is never printed."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env = REPO_ROOT / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("RUNPOD_API_KEY") and "=" in line:
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print(
            "NEEDS_CONTEXT: RUNPOD_API_KEY not found in env or .env — cannot run.",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


# --------------------------------------------------------------------------
# GraphQL: offer discovery + pricing (read-only, free)
# --------------------------------------------------------------------------
def query_offers(client: httpx.Client) -> list[dict]:
    """Return the target GPUs that have 2-GPU COMMUNITY stock right now, each
    as {id, display, price_per_hr (2-GPU total), stock}, sorted cheapest-first.
    price is the on-demand ("uninterruptable") community price for GPU_COUNT."""
    q = """
    query {
      gpuTypes {
        id displayName memoryInGb communityCloud communityPrice
        lowestPrice(input: {gpuCount: %d, secureCloud: false}) {
          uninterruptablePrice stockStatus
        }
      }
    }
    """ % GPU_COUNT
    resp = client.post(GQL_URL, json={"query": q}, timeout=40)
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL pricing query failed: {data['errors']}")
    by_id = {g["id"]: g for g in data["data"]["gpuTypes"]}
    offers = []
    for gid in TARGET_GPUS:
        g = by_id.get(gid)
        if not g or not g.get("communityCloud"):
            continue
        lp = g.get("lowestPrice") or {}
        price = lp.get("uninterruptablePrice")
        stock = lp.get("stockStatus")
        # null price / null stock == no 2-GPU community capacity for this GPU now
        if price is None or stock is None:
            continue
        offers.append(
            {
                "id": gid,
                "display": g.get("displayName", gid),
                "price_per_hr": float(price),
                "per_gpu": g.get("communityPrice"),
                "stock": stock,
                "memory_gb": g.get("memoryInGb"),
            }
        )
    offers.sort(key=lambda o: o["price_per_hr"])
    return offers


# --------------------------------------------------------------------------
# REST: pod lifecycle
# --------------------------------------------------------------------------
def create_pod(client: httpx.Client, name: str, gpu_ids: list[str], pubkey: str) -> dict:
    body = {
        "name": name,
        "imageName": IMAGE,
        "cloudType": "COMMUNITY",
        "computeType": "GPU",
        "gpuTypeIds": gpu_ids,  # priority list; RunPod takes the first available
        "gpuCount": GPU_COUNT,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
        "containerDiskInGb": 25,
        "volumeInGb": 0,  # no persistent volume — nothing to keep
        "env": {"PUBLIC_KEY": pubkey, "SSH_PUBLIC_KEY": pubkey},
    }
    resp = client.post(f"{REST_URL}/pods", json=body, timeout=90)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"pod create failed HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


def get_pod(client: httpx.Client, pod_id: str) -> dict:
    resp = client.get(f"{REST_URL}/pods/{pod_id}", timeout=40)
    resp.raise_for_status()
    return resp.json()


def list_pods(client: httpx.Client) -> list[dict]:
    resp = client.get(f"{REST_URL}/pods", timeout=40)
    resp.raise_for_status()
    data = resp.json()
    # the endpoint returns a list (or {"pods":[...]} on some versions)
    return data if isinstance(data, list) else data.get("pods", data.get("data", []))


def delete_pod(client: httpx.Client, pod_id: str) -> int:
    resp = client.delete(f"{REST_URL}/pods/{pod_id}", timeout=40)
    return resp.status_code


def sweep_orphans(client: httpx.Client, log: Runlog, keep: str | None = None) -> list[str]:
    """Delete every flashruntime-e2e-* pod (optionally except `keep`)."""
    swept = []
    try:
        pods = list_pods(client)
    except Exception as exc:  # noqa: BLE001
        log(f"orphan sweep: list failed ({exc}) — skipping")
        return swept
    for p in pods:
        pid, pname = p.get("id"), p.get("name", "")
        if pid and pid != keep and str(pname).startswith(NAME_PREFIX):
            code = delete_pod(client, pid)
            log(f"orphan sweep: deleted {pname} ({pid}) -> HTTP {code}")
            swept.append(pid)
    return swept


def ssh_endpoint(pod: dict) -> tuple[str | None, int | None]:
    """(public_ip, ssh_port) from a pod object, or (None, None) if not ready."""
    ip = pod.get("publicIp")
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") if isinstance(mappings, dict) else None
    return (ip if ip else None, int(port) if port else None)


# --------------------------------------------------------------------------
# SSH / SCP helpers
# --------------------------------------------------------------------------
def _ssh_base(key: Path, known_hosts: Path) -> list[str]:
    return ["-i", str(key), "-o", f"UserKnownHostsFile={known_hosts}", *SSH_COMMON]


def ssh_run(ip: str, port: int, key: Path, known_hosts: Path, cmd: str, timeout: int):
    argv = [
        "ssh", *_ssh_base(key, known_hosts), "-p", str(port), f"root@{ip}",
        f"bash -lc {shlex.quote(cmd)}",
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def scp_up(ip, port, key, known_hosts, local, remote, timeout):
    argv = ["scp", *_ssh_base(key, known_hosts), "-P", str(port), "-r", str(local), f"root@{ip}:{remote}"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def scp_down(ip, port, key, known_hosts, remote, local, timeout):
    argv = ["scp", *_ssh_base(key, known_hosts), "-P", str(port), "-r", f"root@{ip}:{remote}", str(local)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------
# wheel build (prep, before any paid pod time)
# --------------------------------------------------------------------------
def newest_wheel() -> Path | None:
    wheels = sorted((REPO_ROOT / "dist").glob("flashruntime-*.whl"), key=lambda p: p.stat().st_mtime)
    return wheels[-1] if wheels else None


def build_wheel(log: Runlog) -> Path:
    log("building docs (build_docs.py) then wheel (python -m build) ...")
    subprocess.run([sys.executable, "scripts/build_docs.py"], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "-m", "build", "--wheel"], cwd=REPO_ROOT, check=True)
    wheel = newest_wheel()
    if not wheel:
        raise RuntimeError("wheel build produced no dist/flashruntime-*.whl")
    log(f"wheel: {wheel.name}")
    return wheel


# --------------------------------------------------------------------------
# the on-pod run (returns a dict of captured outputs + pass/fail per step)
# --------------------------------------------------------------------------
REMOTE = "/root/work"


def run_on_pod(ip, port, key, known_hosts, wheel: Path, deadline_at: float, log: Runlog, out_dir: Path) -> dict:
    def budget(cap: int) -> int:
        remaining = int(deadline_at - time.monotonic())
        if remaining < 30:
            raise TimeoutError("deadline reached before phase could start")
        return min(cap, remaining)

    captured: dict = {}

    # 1) wait for sshd, then a preflight (torch + GPU count visible)
    log("waiting for sshd + preflight (torch/GPU visibility) ...")
    deadline_ssh = min(time.monotonic() + SSH_WAIT_BUDGET_S, deadline_at)
    pre = None
    while time.monotonic() < deadline_ssh:
        try:
            pre = ssh_run(ip, port, key, known_hosts,
                          "which python; python -c 'import torch;print(\"torch\",torch.__version__,\"gpus\",torch.cuda.device_count(),\"cuda\",torch.version.cuda)'",
                          timeout=budget(40))
            if pre.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            pass
        time.sleep(8)
    if not pre or pre.returncode != 0:
        raise RuntimeError(f"preflight/sshd never succeeded:\n{(pre.stderr if pre else '')[:400]}")
    log("preflight: " + pre.stdout.strip().replace("\n", " | "))
    captured["preflight"] = (pre.stdout + pre.stderr)

    # 2) upload wheel + examples/ + benchmarks/ + tests/ (layout: siblings under /root/work
    #    so benchmarks/_util EXAMPLES = REPO_ROOT/examples resolves, and the test's
    #    Path(__file__).parent.parent/"examples" resolves to /root/work/examples)
    log("uploading wheel + examples/ + benchmarks/ + tests/ ...")
    mk = ssh_run(ip, port, key, known_hosts, f"mkdir -p {REMOTE}/tests {REMOTE}/results", timeout=budget(40))
    if mk.returncode != 0:
        raise RuntimeError(f"remote mkdir failed:\n{mk.stderr[:300]}")
    for local, remote in [
        (wheel, f"{REMOTE}/"),
        (REPO_ROOT / "examples", f"{REMOTE}/"),
        (REPO_ROOT / "benchmarks", f"{REMOTE}/"),
        (REPO_ROOT / "tests" / "test_gpu_e2e.py", f"{REMOTE}/tests/"),
        (REPO_ROOT / "tests" / "conftest.py", f"{REMOTE}/tests/"),
    ]:
        r = scp_up(ip, port, key, known_hosts, local, remote, timeout=budget(180))
        if r.returncode != 0:
            raise RuntimeError(f"scp {local} failed:\n{r.stderr[:400]}")

    # 3) install the wheel + pytest (torch already in the image — do NOT reinstall it)
    log("pip install <wheel> pytest (torch untouched) ...")
    inst = ssh_run(ip, port, key, known_hosts,
                   f"set -o pipefail; cd {REMOTE} && python -m pip install --no-input ./{wheel.name} pytest 2>&1 | tail -20",
                   timeout=budget(360))
    captured["pip_install"] = inst.stdout + inst.stderr
    if inst.returncode != 0:
        raise RuntimeError(f"pip install failed:\n{inst.stdout[-800:]}")
    log("pip install OK")

    # 4) the GPU acceptance test
    log("running tests/test_gpu_e2e.py -v ...")
    pt = ssh_run(ip, port, key, known_hosts,
                 f"set -o pipefail; cd {REMOTE} && python -m pytest tests/test_gpu_e2e.py -v 2>&1 | tee results/pytest_gpu.txt",
                 timeout=budget(420))
    captured["pytest"] = pt.stdout + pt.stderr
    captured["pytest_rc"] = pt.returncode
    (out_dir / "pytest_gpu.txt").write_text(captured["pytest"])
    log(f"pytest returncode={pt.returncode} (tail: {pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else '?'})")

    # 5) the two GPU benchmark scenarios (run separately — the CLI's --scenario
    #    takes ONE; two invocations is the simplest correct way to get both)
    captured["bench_rc"] = {}
    for scenario in ("loop_overhead", "recovery_economics"):
        log(f"running benchmark scenario {scenario} (repeats=3) ...")
        bench = ssh_run(ip, port, key, known_hosts,
                        f"set -o pipefail; cd {REMOTE} && python -m benchmarks run --scenario {scenario} --repeats 3 "
                        f"--out results/bench_{scenario}.json 2>&1 | tee results/bench_{scenario}.log",
                        timeout=budget(420))
        captured[f"bench_{scenario}"] = bench.stdout + bench.stderr
        captured["bench_rc"][scenario] = bench.returncode
        (out_dir / f"bench_{scenario}.log").write_text(bench.stdout + bench.stderr)
        log(f"benchmark {scenario} returncode={bench.returncode}")

    # 6) bring the results bundle home
    log("scp results/ bundle back ...")
    dl = scp_down(ip, port, key, known_hosts, f"{REMOTE}/results", out_dir, timeout=budget(120))
    if dl.returncode != 0:
        log(f"WARNING: results scp-down failed: {dl.stderr[:200]}")
    return captured


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan-only", action="store_true",
                        help="query offers, print the chosen GPU/price/est cost, create NOTHING")
    parser.add_argument("--wheel", type=Path, default=None, help="prebuilt wheel to upload (default: build/discover)")
    args = parser.parse_args(argv)

    log = Runlog()
    key = load_api_key()
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"})

    # ---- offer discovery (both modes) ----
    log("querying RunPod GraphQL for 2-GPU community offers ...")
    offers = query_offers(client)
    if not offers:
        log("NO 2-GPU community stock in any target GPU right now.")
        print("BLOCKED: no available 2-GPU community offer among "
              + ", ".join(TARGET_GPUS))
        return 1
    chosen = offers[0]
    est_30min = round(chosen["price_per_hr"] * 0.5, 3)
    print("\n=== OFFER SELECTION (cheapest available 2-GPU community) ===")
    for o in offers:
        mark = "*" if o is chosen else " "
        print(f" {mark} {o['display']:<12} 2xGPU ${o['price_per_hr']:.3f}/hr "
              f"(${o['per_gpu']}/gpu) stock={o['stock']} mem={o['memory_gb']}GB")
    print(f"\n CHOSEN : {chosen['display']}  ({chosen['id']})")
    print(f" IMAGE  : {IMAGE}")
    print(f" PRICE  : ${chosen['price_per_hr']:.3f}/hr for {GPU_COUNT} GPUs")
    print(f" EST    : ${est_30min} for the 30-min cap (real runs finish in minutes)")
    print(f" CAPS   : 1 pod, {DEADLINE_S//60}-min hard deadline, terminate-on-any-failure\n")

    if args.plan_only:
        print("--plan-only: created NOTHING. Re-run without --plan-only to execute.")
        return 0

    # ---- prep the wheel BEFORE spending any pod time ----
    wheel = args.wheel or newest_wheel() or build_wheel(log)
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")
    log(f"using wheel {wheel.name}")

    # ---- ephemeral SSH keypair (nothing touches ~/.ssh) ----
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    keydir = out_dir / "_ssh"
    keydir.mkdir(exist_ok=True)
    keyfile = keydir / "id_ed25519"
    known_hosts = keydir / "known_hosts"
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(keyfile),
                    "-C", "flashruntime-e2e"], check=True)
    pubkey = (keydir / "id_ed25519.pub").read_text().strip()
    (out_dir / "plan.json").write_text(json.dumps({"chosen": chosen, "offers": offers, "image": IMAGE}, indent=2))

    # ---- one-at-a-time: sweep any pre-existing labeled pod first ----
    sweep_orphans(client, log)

    pod_id = None
    result_status = "UNKNOWN"
    captured: dict = {}
    price = chosen["price_per_hr"]
    # The provisioned GPU can differ from the cheapest OFFER: RunPod walks the
    # gpuTypeIds priority list and takes the first with live 2-GPU capacity, so
    # a "Low"-stock 3090 commonly falls through to the next entry. Report what we
    # ACTUALLY got (from the create response), not what we asked for first.
    actual_gpu = chosen["display"]
    create_mono = None
    name = f"{NAME_PREFIX}{ts}"
    try:
        # ---- create ----
        gpu_ids = [o["id"] for o in offers]  # priority list, cheapest-first
        log(f"creating pod {name} (gpuTypeIds priority={gpu_ids}) ...")
        pod = create_pod(client, name, gpu_ids, pubkey)
        pod_id = pod.get("id")
        create_mono = time.monotonic()
        deadline_at = create_mono + DEADLINE_S
        if pod.get("costPerHr"):
            price = float(pod["costPerHr"])
        actual_gpu = (pod.get("machine") or {}).get("gpuTypeId") or actual_gpu
        log(f"pod created: id={pod_id} gpu={actual_gpu} costPerHr=${price}/hr  BILLING STARTS NOW")
        (out_dir / "pod_create.json").write_text(json.dumps(pod, indent=2, default=str))

        # ---- poll to RUNNING + public IP + port 22 ----
        log("polling for RUNNING + public IP + port 22 ...")
        ip = port = None
        prov_deadline = min(create_mono + PROVISION_BUDGET_S, deadline_at)
        while time.monotonic() < prov_deadline:
            cur = get_pod(client, pod_id)
            status = cur.get("desiredStatus")
            ip, port = ssh_endpoint(cur)
            if status == "RUNNING" and ip and port:
                log(f"pod RUNNING at {ip}:{port} (gpu={cur.get('machine',{}).get('gpuType') or cur.get('gpu')})")
                (out_dir / "pod_running.json").write_text(json.dumps(cur, indent=2, default=str))
                break
            log(f"  status={status} ip={ip} port={port} ...")
            time.sleep(12)
        else:
            raise TimeoutError("pod never reached RUNNING with SSH endpoint within provisioning budget")

        # ---- do the work ----
        captured = run_on_pod(ip, port, keyfile, known_hosts, wheel, deadline_at, log, out_dir)

        pytest_ok = captured.get("pytest_rc") == 0
        bench_ok = all(rc == 0 for rc in captured.get("bench_rc", {}).values())
        result_status = "SUCCESS" if (pytest_ok and bench_ok) else "PARTIAL/FAIL"
        log(f"work complete: pytest_ok={pytest_ok} bench_ok={bench_ok} -> {result_status}")

    except Exception as exc:  # noqa: BLE001 — any failure aborts to teardown
        result_status = f"ABORTED: {type(exc).__name__}: {exc}"
        log(f"ABORT: {result_status}")
    finally:
        # ---- ALWAYS terminate + sweep ----
        runtime_s = (time.monotonic() - create_mono) if create_mono else 0.0
        if pod_id:
            code = delete_pod(client, pod_id)
            log(f"terminate pod {pod_id} -> HTTP {code}")
        swept = sweep_orphans(client, log)
        # confirm nothing labeled remains
        try:
            remaining = [p for p in list_pods(client) if str(p.get("name", "")).startswith(NAME_PREFIX)]
        except Exception:  # noqa: BLE001
            remaining = ["<list failed>"]
        cost = round(runtime_s / 3600.0 * price, 4)
        termination_proof = (
            f"post-run list: {len(remaining)} flashruntime-e2e-* pod(s) remain "
            f"(swept {len(swept)} straggler(s))"
        )
        log(termination_proof)

        summary = "\n".join([
            "=" * 64,
            "RUNPOD GPU E2E — SUMMARY",
            "=" * 64,
            f"status        : {result_status}",
            f"gpu           : {actual_gpu} x{GPU_COUNT} (selected offer: {chosen['display']})",
            f"image         : {IMAGE}",
            f"price         : ${price:.3f}/hr",
            f"pod runtime   : {runtime_s:.1f}s ({runtime_s/60:.1f} min)",
            f"est. cost     : ${cost}",
            f"pytest_rc     : {captured.get('pytest_rc')}",
            f"bench_rc      : {captured.get('bench_rc')}",
            f"termination   : {termination_proof}",
            f"results dir   : {out_dir}",
            "=" * 64,
        ])
        print("\n" + summary + "\n")
        (out_dir / "timeline.txt").write_text(log.text())
        (out_dir / "summary.txt").write_text(summary + "\n")
        client.close()

    return 0 if result_status == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
