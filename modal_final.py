"""Modal wrappers for the final model: data generation (CPU) and training (one H100). The work itself is in
scripts/gen_final.py and majproj/train_final.py; this file adds containers, the Volume and the spending guard.

Spending rules this file enforces:
  every launch goes through `guard`, which reads the billing report, adds the launch estimate and the estimates of
  launches that may still be running or unbilled, writes the check to logs/modal_budget.log and refuses above LIMIT_USD;
  every function has an explicit timeout, retries=0, max_containers=1, a 2 s scale-down and no schedule;
  each launch shortens the function timeout to its own estimate through with_options.

Commands (after activating nothing: this uses the active profile):
  uv run modal run modal_final.py::gen --split val  --shards 8
  uv run modal run modal_final.py::gen --split test --shards 8
  uv run modal run modal_final.py::gen --split train --shards 200
  uv run modal run modal_final.py::manifest
  uv run modal run --detach modal_final.py::fit --run-id sim1 --wall-min 170
  uv run modal run --detach modal_final.py::fit --run-id real1 --wall-min 60 --real --init-from /vol/runs/final/sim1/final.pt
  uv run modal volume get majproj-vol runs/final/sim1/final.pt checkpoints/final_sim1.pt --force
  uv run modal volume get majproj-vol runs/final/sim1/train_final.jsonl logs/train_final.jsonl --force
  uv run modal app stop majproj-final
"""
import datetime as dt
import json
import os
import subprocess
import time

import modal

LIMIT_USD = 24.00
H100_USD_H = 3.95
CORE_USD_H = 0.047
GIB_USD_H = 0.008
LEDGER = f"logs/modal_budget{'.' + os.environ['MODAL_PROFILE'] if os.environ.get('MODAL_PROFILE') else ''}.log"   # one per account: billing is per account
BILLING_LAG_S = 1200

GEN_CPU, GEN_MEM_MB, GEN_TIMEOUT_S = 16, 16384, 1800
TRAIN_CPU, TRAIN_MEM_MB = 8, 32768
# The deployed function's hard timeout: one leg's wall limit plus the margin. Set MAJPROJ_WALL_MIN at deploy time.
TRAIN_TIMEOUT_S = int(float(os.environ.get('MAJPROJ_WALL_MIN', '180')) * 60) + 600
TIMEOUT_MARGIN_S = 600

app = modal.App('majproj-final')
vol = modal.Volume.from_name('majproj-vol', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.11')
    .uv_pip_install('torch==2.14.1', 'numpy==2.4.6', 'scipy==1.17.1', 'easydict==1.13', 'einops==0.8.2', 'pyyaml==6.0.3')
    .env({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'})
    .add_local_python_source('majproj', 'analysis')
    .add_local_file('scripts/gen_final.py', '/root/scripts/gen_final.py')
    .add_local_file('scripts/run_pipeline.py', '/root/scripts/run_pipeline.py')
    .add_local_file('configs/final.yaml', '/root/configs/final.yaml')
    .add_local_file('configs/real_bank.npz', '/root/configs/real_bank.npz')      # the 'real' regime of majproj.sim_raw
    .add_local_file('configs/real_bank.json', '/root/configs/real_bank.json')
    .add_local_file('configs/real_bottle_parts.json', '/root/configs/real_bottle_parts.json')   # majproj.real_targets
    .add_local_file('configs/real_box_parts.json', '/root/configs/real_box_parts.json')
    .add_local_file('checkpoints/AdaPoinTr_PCN.pth', '/ckpt/AdaPoinTr_PCN.pth')
    .add_local_dir('FINAL-INPUTS', '/real')
)
LIMITS = dict(retries=0, max_containers=1, scaledown_window=2, image=image, volumes={'/vol': vol})


def _config():
    from scripts.gen_final import code_hash, load_config
    cfg, cfg_hash = load_config()
    return cfg, dict(code=code_hash(), config=cfg_hash)


@app.function(cpu=GEN_CPU, memory=GEN_MEM_MB, timeout=GEN_TIMEOUT_S, **LIMITS)
def gen_shard(split: str, shard: int, per_shard: int | None = None):
    from scripts.gen_final import gen_shard_file
    cfg, _ = _config()
    report = gen_shard_file(f"/vol/{cfg['gen']['root']}", split, shard, cfg, cfg['gen']['workers'], per_shard)
    vol.commit()
    return report


@app.function(cpu=2, memory=2048, timeout=300, **LIMITS)
def write_manifest():
    from scripts.gen_final import write_manifest as write
    cfg, hashes = _config()
    vol.reload()
    return write(f"/vol/{cfg['gen']['root']}", hashes['config'])


@app.function(gpu=os.environ.get('MAJPROJ_GPU', 'H100'), cpu=TRAIN_CPU, memory=TRAIN_MEM_MB, timeout=TRAIN_TIMEOUT_S, **LIMITS)
def train(run_id: str, wall_limit_s: float, real: bool = False, init_from: str | None = None,
          resume_from: str | None = None, overrides: dict | None = None, force: bool = False):
    from majproj.train_final import run
    cfg, hashes = _config()
    cfg['real']['enabled'] = real
    for section, values in (overrides or {}).items():
        cfg[section].update(values)
    vol.reload()
    return run(cfg, hashes, f"/vol/{cfg['gen']['root']}", '/vol/runs/final', run_id, wall_limit_s, '/ckpt/AdaPoinTr_PCN.pth',
               'cuda', input_root='/real', init_from=init_from, resume_from=resume_from, force=force, commit=vol.commit)


# ------------------------------------------------------------------------------------------------- local side

def _billed_this_month():
    r = subprocess.run(['uv', 'run', 'modal', 'billing', 'report', '--for', 'this month', '--json'], capture_output=True,
                       text=True, check=True, timeout=120)
    return sum(float(row['cost']) for row in json.loads(r.stdout))


def guard(what, estimate_usd, run_s=0, billed=None, ledger=LEDGER, now=None):
    """Refuses a launch that could pass LIMIT_USD. An earlier launch counts at its full estimate until `run_s` plus
    BILLING_LAG_S after it started, since it may still be running and billing lags. That can count a job twice, which
    errs on the safe side."""
    billed = _billed_this_month() if billed is None else billed
    now = time.time() if now is None else now
    recent = 0.0
    if os.path.exists(ledger):
        with open(ledger) as f:
            rows = [json.loads(l) for l in f if l.startswith('{')]
        recent = sum(r['estimate_usd'] for r in rows if r['launched'] and now < r['ts'] + r['run_s'] + BILLING_LAG_S)
    total = billed + recent + estimate_usd
    ok = total <= LIMIT_USD
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    with open(ledger, 'a') as f:
        f.write(json.dumps(dict(ts=now, utc=dt.datetime.fromtimestamp(now, dt.UTC).isoformat(timespec='seconds'), what=what, run_s=run_s,
                                billed_usd=round(billed, 4), unbilled_usd=round(recent, 4), estimate_usd=round(estimate_usd, 4),
                                projected_usd=round(total, 4), limit_usd=LIMIT_USD, launched=ok)) + '\n')
    if not ok:
        raise SystemExit(f'budget guard: {what} refused: billed {billed:.2f} + unbilled {recent:.2f} + estimate '
                         f'{estimate_usd:.2f} = {total:.2f} > {LIMIT_USD:.2f}')
    print(f'[budget] {what}: billed {billed:.2f} + unbilled {recent:.2f} + estimate {estimate_usd:.2f} = {total:.2f} '
          f'of {LIMIT_USD:.2f}', flush=True)


def gen_cost(hours):
    return hours * (GEN_CPU * CORE_USD_H + GEN_MEM_MB / 1024 * GIB_USD_H)


def train_cost(hours, mem_mb=TRAIN_MEM_MB):
    return hours * (H100_USD_H + TRAIN_CPU * CORE_USD_H + mem_mb / 1024 * GIB_USD_H)


@app.local_entrypoint()
def gen(split: str = 'train', shards: int = 8, start: int = 0, per_shard: int = 0, max_usd: float = 2.0):
    """Shards run one after another. Stops by itself once the measured runtime has cost `max_usd`."""
    t0 = time.time()
    overrun = gen_cost(GEN_TIMEOUT_S / 3600)                  # one shard may finish after the limit is reached
    guard(f'gen {split} {start}+{shards}', max_usd + overrun, run_s=int(max_usd / gen_cost(1) * 3600) + GEN_TIMEOUT_S)
    for i in range(start, start + shards):
        spent = gen_cost((time.time() - t0) / 3600)
        if spent >= max_usd:
            print(f'[gen] stopped before shard {i}: about ${spent:.2f} spent, limit ${max_usd:.2f}', flush=True)
            break
        r = gen_shard.remote(split, i, per_shard or None)
        print(json.dumps(r), flush=True)
    print(f'[gen] about ${gen_cost((time.time() - t0) / 3600):.2f} for this launch', flush=True)


@app.local_entrypoint()
def manifest():
    guard('manifest', 0.01, run_s=300)
    m = write_manifest.remote()
    print(json.dumps(m, indent=1)[:4000])


def spawn_leg(run_id: str, wall_min: float, real: bool, init_from: str, resume_from: str, overrides: dict, force: bool):
    """Start a training leg on the DEPLOYED app (`uv run modal deploy modal_final.py` first, with MAJPROJ_WALL_MIN set
    to the same wall_min). The call runs on Modal with no local process attached, so a dead laptop session cannot
    stop it. The deployed function's own timeout is the cap, so the guard uses it as the worst case."""
    fn = modal.Function.from_name('majproj-final', 'train')
    timeout_s = TRAIN_TIMEOUT_S
    if timeout_s < wall_min * 60 + TIMEOUT_MARGIN_S:
        raise SystemExit(f'deployed timeout {timeout_s} s is shorter than wall {wall_min} min + margin: redeploy with MAJPROJ_WALL_MIN={wall_min:g}')
    guard(f'train {run_id}', train_cost(timeout_s / 3600), run_s=timeout_s)
    call = fn.spawn(run_id, wall_min * 60, real, init_from or None, resume_from or None, overrides, force)
    print(json.dumps(dict(run_id=run_id, call_id=call.object_id, wall_min=wall_min, timeout_s=timeout_s)), flush=True)
    return call.object_id


@app.local_entrypoint()
def fit(run_id: str, wall_min: float, real: bool = False, init_from: str = '', resume_from: str = '',
        overrides: str = '{}', memory_mb: int = TRAIN_MEM_MB, force: bool = False):
    """`force` resumes a checkpoint whose code hash differs (a deliberate code change between stages); the run log
    records the mismatch."""
    timeout_s = int(wall_min * 60 + TIMEOUT_MARGIN_S)
    guard(f'train {run_id}', train_cost(timeout_s / 3600, memory_mb), run_s=timeout_s)      # worst case: the whole timeout
    fn = train.with_options(timeout=timeout_s, memory=memory_mb)
    print(json.dumps(fn.remote(run_id, wall_min * 60, real, init_from or None, resume_from or None, json.loads(overrides), force),
                     indent=1)[:4000])
