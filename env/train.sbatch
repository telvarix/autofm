#!/bin/bash
# ==========================================================================
# autofm - Leonardo (CINECA) SLURM job template
# ==========================================================================
# See docs/ENVIRONMENT_SETUP.md §4.
#
# Before submitting:
#   1. Replace <ACCOUNT>, <SHARED_ENV_PATH>, <PROJECT_PATH> with your values
#      (or source .env and substitute).
#   2. Choose debug vs full run (see QoS note below).
#   3. Adjust --gres for GPU count.
#
# Submit:   sbatch env/train.sbatch
# Monitor:  squeue --me
# Cancel:   scancel <jobid>
# Logs:     logs/autofm-train-<jobid>.out / .err
# ==========================================================================

# --- SLURM directives ------------------------------------------------------
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg       # debug queue: max 30 min. REMOVE for real runs (max 24 h).
#SBATCH --job-name=autofm-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # 1 A100. Set gpu:4 for whole node.
#SBATCH --time=00:30:00           # keep within QoS limit; raise for real runs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Setup ------------------------------------------------------------------
mkdir -p logs

# Activate the shared team environment
source <SHARED_ENV_PATH>/activate.sh

# Sanity: confirm the GPU is visible AND that the torch CUDA build matches the
# node driver. A mismatch (torch built for a newer CUDA than the driver
# supports) is the usual cause of NCCL "CUDA driver version is insufficient"
# crashes during DDP init - surfacing both numbers here makes it obvious.
echo "-------- GPU visibility --------"
echo "[driver] nvidia-smi reports (top-right = max CUDA the driver supports):"
nvidia-smi | head -n 4 || echo "nvidia-smi unavailable (login node?)"
python -c "
import torch
print('torch        ', torch.__version__)
print('torch CUDA   ', torch.version.cuda, '  <- torch was BUILT for this CUDA')
print('CUDA avail   ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device       ', torch.cuda.get_device_name(0))
    print('GPU memory   ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
else:
    print('WARNING: CUDA not available. If torch CUDA above is newer than the')
    print('         driver CUDA from nvidia-smi, reinstall a matching build, e.g.:')
    print('           pip install --force-reinstall torch \\\\')
    print('             --index-url https://download.pytorch.org/whl/cu121')
"
echo "---------------------------"

# --- Run --------------------------------------------------------------------
cd <PROJECT_PATH>/<your-project-dir>

# Single-GPU training (default)
python -m autofm.train_mae --config config_autofm.yaml

# --- Multi-GPU (whole node, 4x A100) --------------------------------------
# Uncomment below and set --gres=gpu:4 above:
#   torchrun --nproc_per_node=4 -m autofm.train_mae --config config_autofm.yaml
# Get single-GPU correct first.
