# autofm - Environment Setup Guide

Step-by-step setup for the two environments this project runs in, plus
verification, a runnable smoke test, and a troubleshooting reference.

| Environment | Role | Hardware | PyTorch build |
|-------------|------|----------|---------------|
| **Local** | Development, data prep, small smoke-test runs | CPU (or Apple Silicon / MPS) | CPU build |
| **Leonardo (CINECA)** | Real training, HPO, scale-out | NVIDIA A100 (CUDA 12.x) | CUDA build (shared env) |

> **Golden rule - match the team's versions.** The shared Leonardo environment
> is the source of truth for package versions (`env/requirements.lock` on the
> cluster). Your local machine should match it so a model behaves the same in
> both places. The only intended difference is the PyTorch build: a CPU (or MPS)
> build locally, a CUDA build on Leonardo. Everything local runs inside a
> **virtual environment (venv)** - never the system Python.

---

## Table of contents

1. [Before you start](#1-before-you-start)
2. [Version matrix](#2-version-matrix)
3. [Local setup (CPU / MPS)](#3-local-setup-cpu--mps)
4. [Verify the local install](#4-verify-the-local-install)
5. [Run the smoke test](#5-run-the-smoke-test)
6. [Leonardo (CINECA) - the shared environment](#6-leonardo-cineca---the-shared-environment)
7. [Running training on Leonardo (SLURM)](#7-running-training-on-leonardo-slurm)
8. [Config files: smoke vs full scale](#8-config-files-smoke-vs-full-scale)
9. [Leonardo gotchas](#9-leonardo-gotchas-read-before-your-first-job)
10. [Keeping the two environments aligned](#10-keeping-the-two-environments-aligned)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick reference](#12-quick-reference)

---

## 1. Before you start

**What this gets you:** a working `autofm` install you can develop against
locally and a clear path to real GPU training on Leonardo.

**Prerequisites**

| Requirement | Local | Leonardo |
|-------------|-------|----------|
| Python 3.14+ | you install | provided by shared env |
| Git + the cloned repo | yes | yes |
| NVIDIA GPU / CUDA | no (CPU/MPS is fine) | provided (A100) |
| Cluster account + allocation | - | yes (from your team lead) |

**Time estimate:** local setup is ~5-10 minutes; Leonardo needs no build (you
use the shared env).

**Where things live**

| Artifact | Path | Purpose |
|----------|------|---------|
| Direct dependencies | `env/requirements.in` | edit to add a package |
| Resolved pins | `env/requirements.lock` | reproducible install (from Leonardo) |
| Install entry point | `requirements.txt` | installs `-r env/requirements.in` |
| SLURM template | `env/train.sbatch` | cluster job template |
| Cluster values | `env/.env.example` -> copy to `.env` | SLURM account, paths (git-ignored) |
| Smoke config | `config_autofm.yaml` | small local/dev run |
| Full-scale config | `config_leonardo.yaml` | A100 training run |

---

## 2. Version matrix

These mirror the shared Leonardo environment; `env/requirements.lock` on the
cluster is the authoritative pinned list.

| Component | Version | Notes |
|-----------|---------|-------|
| Python | **3.14+** | Matches the shared Leonardo env. |
| PyTorch (`torch`) | per shared env | Leonardo: CUDA build. Local: CPU/MPS build. |
| PyTorch Geometric | matching `torch` | Pure-Python core; in the shared env. |
| numpy, pandas, scipy | per `requirements.lock` | Core compute. |
| scikit-learn | per `requirements.lock` | Baselines + metrics. |
| xgboost-cpu | per `requirements.lock` | **CPU-only build** ([§9](#9-leonardo-gotchas-read-before-your-first-job)). |
| pyarrow | per `requirements.lock` | Reads the Parquet copy of the dataset. |
| mapie | per `requirements.lock` | Conformal prediction. |
| pyyaml | per `requirements.lock` | Config loading. |
| matplotlib, networkx | per `requirements.lock` | Plots + graphs. |
| optuna, tensorboard | per `requirements.lock` | HPO + monitoring. |

The direct (loosely pinned) list is `env/requirements.in`; the exact resolved
versions are `env/requirements.lock`. When the two disagree, the lock wins.

---

## 3. Local setup (CPU / MPS)

If your machine has **no NVIDIA GPU / CUDA**, install the **CPU build** of
PyTorch (Apple Silicon additionally gets the MPS backend automatically). Local
is for development, data preparation, and small smoke-test runs; real GPU
training happens on Leonardo.

### 3.1 Check Python

```bash
python3 --version          # need 3.14+ to match the shared Leonardo env
```

Install Python 3.14+ via your OS package manager if needed.

### 3.2 Create and activate the venv

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
which python               # should end in .venv/bin/python
python --version           # should be 3.14+
python -m pip install --upgrade pip setuptools wheel
```

### 3.3 Install PyTorch + libraries

```bash
pip install torch                 # CPU/MPS build (use the official index URL if your OS needs it)
pip install -r requirements.txt   # installs env/requirements.in
```

> **To match Leonardo exactly:** copy `env/requirements.lock` from the cluster
> and `pip install -r env/requirements.lock` locally (dropping any CUDA-only
> pins). The closer the versions, the fewer "works on my machine" surprises.

---

## 4. Verify the local install

```bash
python - <<'PY'
import torch, torch_geometric, sklearn, xgboost, yaml
print("torch        ", torch.__version__)
print("cuda available", torch.cuda.is_available())     # False locally - expected
print("mps  available", getattr(torch.backends, "mps", None)
      and torch.backends.mps.is_available())            # True on Apple Silicon
print("pyg          ", torch_geometric.__version__)
PY
```

Expected locally: `cuda available False`. On Apple Silicon `mps available True`.
The pipeline selects the device at runtime (`cuda` -> `mps` -> `cpu`), so the
same code runs unchanged everywhere.

---

## 5. Run the smoke test

Confirm the whole pipeline runs end to end on a small slice. This needs a
dataset at the path in `config_autofm.yaml` (`dataset_root`); point it at your
local copy first.

```bash
# End-to-end FM pipeline on the smoke config (small max_days / max_gnbs)
python -m autofm.pipeline --config config_autofm.yaml

# Just the MAE pretraining stage
python -m autofm.train_mae --config config_autofm.yaml

# Profile a dataset export (regenerates docs/DATA_NOTES.md)
python -m autofm.profile_dataset
```

The pipeline prints a 5-stage progress log and a final `TEST macro-F1` line.
Runnable module entry points in this repo: `autofm.pipeline`,
`autofm.train_mae`, `autofm.profile_dataset`, `autofm.bench`,
`autofm.advisory.demo`.

> **Override config knobs from the CLI** without editing the file:
> `python -m autofm.pipeline --config config_autofm.yaml --max-days 2 --max-gnbs 4 --compressor pca`.

---

## 6. Leonardo (CINECA) - the shared environment

**Do not build your own environment on Leonardo.** The team already has a
shared, tested environment (Python, PyTorch/CUDA, PyG, and all our libraries),
validated on an A100 node. Use it.

> **Placeholders - replace with your own details.** Substitute your actual
> values (from your team lead / cluster allocation). These also live in
> `env/.env.example`; copy it to `.env` and fill it in.
> - `<ACCOUNT>` -> your SLURM project account.
> - `<SHARED_ENV_PATH>` -> absolute path to the team's shared env directory.
> - `<PROJECT_PATH>` -> your permanent project/work directory on the cluster.

| Fact | Value |
|------|-------|
| Project account | `<ACCOUNT>` |
| Shared env path | `<SHARED_ENV_PATH>` |
| Activate script | `<SHARED_ENV_PATH>/activate.sh` |
| Pinned package list | `<SHARED_ENV_PATH>/requirements.lock` |
| GPU partition | `boost_usr_prod` (A100, up to 4 GPUs/node) |
| Debug QoS | `boost_qos_dbg` (max 30 min) |
| Full-run QoS | default (max 24 h) |
| Permanent storage | `<PROJECT_PATH>` |
| Scratch storage | `$CINECA_SCRATCH` (auto-deleted after 40 days) |

### 6.1 Log in

```bash
ssh <your-username>@login.leonardo.cineca.it
```

### 6.2 Activate the shared environment

In an interactive shell or inside a job script:

```bash
source <SHARED_ENV_PATH>/activate.sh
```

That is the whole setup - no venv to create, no modules to load, no pip
installs.

### 6.3 Quick interactive GPU session (debug queue, 30 min)

```bash
srun -A <ACCOUNT> -p boost_usr_prod -q boost_qos_dbg \
     --gres=gpu:1 -c 8 --time=00:30:00 --pty bash

# then, on the GPU node:
source <SHARED_ENV_PATH>/activate.sh
python - <<'PY'
import torch
print("torch        ", torch.__version__)
print("cuda available", torch.cuda.is_available())    # True here
print("device       ", torch.cuda.get_device_name(0)) # NVIDIA A100
PY
exit
```

> The **first `import torch` can take ~1 minute** (shared filesystem). Normal,
> not a hang.

### 6.4 Need another package?

Do not `pip install` ad hoc into the shared env. Instead:

1. Add the package to `env/requirements.in` (one per line, e.g. `seaborn` or
   `seaborn==0.14.0`).
2. On a **login node**, rebuild the env (a few minutes):
   ```bash
   bash <SHARED_ENV_PATH>/setup_env.sh
   ```
3. Commit **both** `env/requirements.in` and the regenerated
   `env/requirements.lock`.

---

## 7. Running training on Leonardo (SLURM)

Use the repo's `env/train.sbatch` as a template; it already has the right
partition/QoS layout and a GPU/driver sanity check. A job looks like:

```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg       # debug: max 30 min. REMOVE this line for real runs (max 24 h).
#SBATCH --job-name=autofm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # up to 4 A100s per node
#SBATCH --time=00:30:00           # within the debug limit; raise for real runs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# Activate the shared team environment
source <SHARED_ENV_PATH>/activate.sh

# Sanity line in the log
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- training command (full-scale config) ---
cd <PROJECT_PATH>/<your-project-dir>
python -m autofm.train_mae --config config_leonardo.yaml
```

Submit and manage:

```bash
sbatch env/train.sbatch    # submit
squeue --me                # check status
scancel <jobid>            # cancel
```

> **Multi-GPU (whole node, 4x A100):** set `--gres=gpu:4` and launch with
> `torchrun --nproc_per_node=4 -m autofm.train_mae --config config_leonardo.yaml`.
> Get a single-GPU run correct first.

---

## 8. Config files: smoke vs full scale

The project ships two configs at the repo root. Both are read with
`--config`; CLI flags override individual keys.

| Key | `config_autofm.yaml` (smoke) | `config_leonardo.yaml` (full) |
|-----|------------------------------|-------------------------------|
| `dataset_root` | local path (set per machine) | cluster scratch path |
| `max_days` | 14 | 182 |
| `max_gnbs` | 40 | 100 |
| `mae_epochs` / `clf_epochs` | 6 / 12 | 50 / 50 |
| `compressor` | `mae` | `mae` |
| `clf_backbone` | `transformer` | `patchtst` |
| `clf_head` | `softmax` | `evidential` |
| `batch_size` | default | 1024 |
| `horizon` / `window_k` / `latent_dim` | 5 / 32 / 32 | 5 / 32 / 32 |

`dataset_root` is machine-specific and must not be committed; set it per
machine. Model and data hyperparameters belong in these config files, **not** in
`.env` (which is for cluster/account values only).

---

## 9. Leonardo gotchas (read before your first job)

Real constraints of this specific cluster/environment; ignoring them causes most
first-time failures.

- **Compute nodes have NO internet.** Download any data/models on the **login
  node** first. Never put `pip install` or a download inside a job script.
- **Do not train on the login node.** Always use `sbatch` or `srun`.
- **The CUDA driver is fixed by the cluster.** Plain PyTorch (eager mode) works
  fine. If the shared env's `torch` was built for a newer CUDA than the node
  driver supports, DDP init fails with NCCL "CUDA driver version is
  insufficient" - the `train.sbatch` sanity check prints both numbers so a
  mismatch is obvious. The authoritative CUDA version is whatever
  `nvidia-smi` and `torch.version.cuda` report in the shared env; do not
  hard-code a number in your scripts.
  - `torch.compile` may be unavailable -> **use eager mode.**
  - **XGBoost on GPU may be unavailable -> use CPU XGBoost.** Install it as
    `xgboost-cpu`: the plain `xgboost` wheel pulls `nvidia-nccl-cu13`, which
    overwrites the CUDA 12 `libnccl` that torch uses, and multi-GPU runs then
    fail with "CUDA driver version is insufficient".
- **`torch.jit.script` may be unsupported on the shared env's Python.** Ignore
  the warning, or use `torch.compile` where appropriate.
- **`$FAST` is full.** Put data and outputs in `<PROJECT_PATH>` (permanent) or
  `$CINECA_SCRATCH` (auto-deleted after 40 days). Not `$FAST`.
- **First `import torch` ~1 min** (shared filesystem) - expected.

---

## 10. Keeping the two environments aligned

The shared Leonardo env is the source of truth (`env/requirements.lock`). Keep
the local machine close to it; the only legitimate difference is the `torch`
build (CPU/MPS locally vs. CUDA on Leonardo). If a model behaves differently
across the two, version drift is the first thing to check:

```bash
# compare installed versions against the cluster lock
pip freeze | sort > /tmp/local.lock
diff <(sort env/requirements.lock) /tmp/local.lock
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No module named autofm` | not at repo root / venv not active | `cd` to repo root; `source .venv/bin/activate` |
| `No module named autofm.pipeline.train` | **wrong entry point** | use `python -m autofm.pipeline` or `python -m autofm.train_mae` (no `.train` submodule exists) |
| `FileNotFoundError: No pm/ day directories under ...` | `dataset_root` wrong/empty | point `dataset_root` in the config at your dataset copy |
| `cuda available False` on Leonardo GPU node | ran on a login node, or torch/driver mismatch | run via `srun`/`sbatch`; check the sanity block in `train.sbatch` |
| NCCL "CUDA driver version is insufficient" | torch built for newer CUDA than driver | use the shared env's `torch`; do not install your own CUDA build on the cluster |
| First `import torch` hangs ~1 min | shared filesystem warm-up | wait; it is expected on Leonardo |
| Job dies at 30 min | still on `boost_qos_dbg` | remove the debug QoS line and raise `--time` (max 24 h) |
| Package missing in shared env | ad-hoc install not allowed | add to `env/requirements.in`, run `setup_env.sh` on a login node |

---

## 12. Quick reference

| Task | Local (CPU/MPS) | Leonardo (GPU) |
|------|-----------------|----------------|
| Set up | `python3 -m venv .venv` + `pip install -r requirements.txt` | nothing - use the shared env |
| Activate | `source .venv/bin/activate` | `source <SHARED_ENV_PATH>/activate.sh` |
| Install torch | `pip install torch` (CPU/MPS) | already in the shared env |
| Smoke test | `python -m autofm.pipeline --config config_autofm.yaml` | same, via `srun`/`sbatch` |
| Run training | `python -m autofm.train_mae --config config_autofm.yaml` | `sbatch env/train.sbatch` (uses `config_leonardo.yaml`) |
| Interactive GPU | n/a (CPU/MPS only) | `srun -A <ACCOUNT> -p boost_usr_prod -q boost_qos_dbg --gres=gpu:1 -c 8 --time=00:30:00 --pty bash` |
| Add a package | `pip install <pkg>` | add to `env/requirements.in`, run `setup_env.sh` on login node |
| Where to put data | project folder (set `dataset_root`) | `<PROJECT_PATH>` or `$CINECA_SCRATCH` |

---

*Team AFM_Ericsson - European AI Hackathon 2026*
