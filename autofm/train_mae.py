"""DDP-capable MAE training entry point.

Trains the masked autoencoder (the autofm compression stage) with PyTorch
DistributedDataParallel so it scales across the 4x A100 of a Leonardo node,
while still running as a plain single process on a laptop (CPU/MPS) when
launched without ``torchrun``.

Launch:
    # single process (laptop / single GPU)
    python -m autofm.train_mae --config config_autofm.yaml

    # multi-GPU on one node
    torchrun --nproc_per_node=4 -m autofm.train_mae --config config_autofm.yaml

Distribution is driven by the standard ``torchrun`` environment variables
(``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``). When ``WORLD_SIZE`` is unset or 1
we skip the process group entirely, so there is no NCCL/CUDA dependency on a
Mac. Only rank 0 prints, evaluates, and writes the checkpoint; the AMP/NVTX
approach mirrors :meth:`autofm.model.compress.MAECompressor.fit` (CUDA-only),
and each process keeps its own ``GradScaler``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from autofm.data.features import WindowRows, prepare_rows
from autofm.data.labels import label_frame, load_alarms
from autofm.data.loader import load
from autofm.model.compress import _MAENet
from autofm.pipeline import load_config


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
def _dist_info() -> tuple[int, int, int, bool]:
    """Read (rank, local_rank, world_size, is_distributed) from the env.

    ``torchrun`` sets RANK/LOCAL_RANK/WORLD_SIZE. Absent (or WORLD_SIZE=1) we
    run as a single plain process - the laptop path.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size, world_size > 1


def _pick_device(local_rank: int) -> torch.device:
    """cuda:LOCAL_RANK when CUDA is present, else MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _preflight_cuda(is_distributed: bool) -> None:
    """Fail fast with an actionable message on a CUDA runtime/driver mismatch.

    DDP uses the NCCL backend, which raises an opaque multi-rank error
    (``ncclUnhandledCudaError`` / "CUDA driver version is insufficient for CUDA
    runtime version") when the installed torch CUDA *runtime* is newer than the
    node's driver supports. That happens on Leonardo when ``pip install torch``
    pulls the newest CUDA build (e.g. +cu126) onto a node whose driver only
    supports an older CUDA. We surface the real cause here - before NCCL init
    buries it across ranks - with the one-line fix.

    No-op in single-process mode (laptop CPU/MPS path).
    """
    if not is_distributed:
        return
    built_for = torch.version.cuda
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Distributed (torchrun) run requested but CUDA is not available. "
            f"torch {torch.__version__} is built for CUDA {built_for}. On a GPU "
            "node this usually means the torch build is newer than the driver "
            "supports (compare `nvidia-smi` top-right CUDA version). Install a "
            "matching build, e.g. cu121:\n"
            "  pip install --force-reinstall torch "
            "--index-url https://download.pytorch.org/whl/cu121"
        )
    try:
        # Touch the CUDA context so a driver/runtime mismatch surfaces here as a
        # readable error, rather than inside the NCCL collective during DDP init.
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(
            f"CUDA initialization failed: {exc}\n"
            f"torch {torch.__version__} is built for CUDA {built_for}; the node "
            "driver may be older (check `nvidia-smi` top-right CUDA version). "
            "Reinstall a torch build matching the driver, e.g. cu121:\n"
            "  pip install --force-reinstall torch "
            "--index-url https://download.pytorch.org/whl/cu121"
        ) from exc


def _setup(is_distributed: bool, local_rank: int) -> torch.device:
    """Init the process group (NCCL) for DDP; no-op in single-process mode."""
    # Validate the CUDA context first so a runtime/driver mismatch yields a
    # clear message instead of an opaque multi-rank NCCL stack trace.
    # Select this rank's GPU first: otherwise the preflight below opens a
    # context on cuda:0 in every rank.
    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    _preflight_cuda(is_distributed)
    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
    return _pick_device(local_rank)


def _cleanup(is_distributed: bool) -> None:
    if is_distributed:
        torch.distributed.destroy_process_group()


def _write_bench(bench_out, step_ms, local_batch, world_size, is_distributed,
                 device, is_main, log) -> None:
    """Aggregate throughput across ranks and write a JSON on rank 0.

    Each rank measures its own median step time -> local samples/sec
    (``local_batch / median_step``). Total throughput is the sum across ranks:
    with DDP every rank processes ``local_batch`` samples per step in parallel.
    We ``all_reduce(SUM)`` the per-rank rate so the number is correct even if
    ranks run at slightly different speeds; the reported median_step_ms is the
    max (slowest) rank, which is what actually bounds a synchronized step.
    """
    if not step_ms:
        if is_main:
            log("[bench] no timed steps recorded (increase steps/epochs); "
                "skipping bench output")
        return

    local_median_ms = statistics.median(step_ms)
    local_rate = local_batch / (local_median_ms / 1000.0)

    if is_distributed:
        rate_t = torch.tensor([local_rate], dtype=torch.float64, device=device)
        ms_t = torch.tensor([local_median_ms], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(rate_t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ms_t, op=torch.distributed.ReduceOp.MAX)
        total_rate = float(rate_t.item())
        median_ms = float(ms_t.item())
    else:
        # Single process: total throughput == its own rate (world_size is 1).
        total_rate = local_rate * world_size
        median_ms = local_median_ms

    if is_main:
        payload = {
            "world_size": world_size,
            "gpus": world_size,
            "median_step_ms": round(median_ms, 3),
            "samples_per_sec": round(total_rate, 1),
        }
        out_path = Path(bench_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        log(f"[bench] median {median_ms:.2f} ms/step  "
            f"{total_rate:.0f} samples/s total across {world_size} rank(s)  "
            f"-> {out_path}")


class _WindowStepRows(Dataset):
    """The flattened train windows, served from the per-ROP rows.

    Item ``j`` is step ``j % k`` of window ``j // k`` - the same row, in the same
    order, as ``windows().reshape(-1, F)[j]`` - without materialising the ``k``
    overlapping copies. ``__getitems__`` lets the DataLoader fetch a whole
    batch with one fancy index instead of one row at a time.
    """

    def __init__(self, rows: np.ndarray, starts: np.ndarray, k: int):
        self.rows, self.starts, self.k = rows, starts, k

    def __len__(self) -> int:
        return len(self.starts) * self.k

    def _row(self, j):
        j = np.asarray(j, dtype=np.int64)
        return self.starts[j // self.k] + j % self.k

    def __getitem__(self, j):
        return torch.from_numpy(self.rows[self._row(j)])

    def __getitems__(self, js):
        return torch.from_numpy(self.rows[self._row(js)])


def _as_batch(x: torch.Tensor) -> tuple[torch.Tensor]:
    """collate_fn: ``__getitems__`` already returns the stacked batch."""
    return (x,)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def fit_mae_rows(cfg: dict, train_rows: WindowRows, device: torch.device,
                 ctx: tuple, log, bench_out: str | None = None) -> _MAENet:
    """Train the MAE on every step of every train window, read through the
    per-ROP rows. DDP when ``ctx`` says so; returns the trained core network.

    ``ctx`` is ``(rank, local_rank, world_size, is_distributed, is_cuda)``.
    """
    rank, local_rank, world_size, is_distributed, is_cuda = ctx
    is_main = rank == 0
    dataset = _WindowStepRows(train_rows.rows, train_rows.starts, train_rows.k)
    in_dim = train_rows.rows.shape[1]
    log(f"[data] train rows {len(dataset)} ({len(train_rows.rows)} distinct), "
        f"feature dim {in_dim}")

    # ---- Model + DDP -------------------------------------------------------
    net = _MAENet(in_dim, cfg["latent_dim"]).to(device)
    if is_distributed:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("mae_lr", 1e-3))
    loss_fn = nn.MSELoss()

    use_amp = bool(cfg.get("amp", True)) and is_cuda
    pin = is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _push(t):
        if is_cuda:
            torch.cuda.nvtx.range_push(t)

    def _pop():
        if is_cuda:
            torch.cuda.nvtx.range_pop()

    # ---- DataLoader + DistributedSampler -----------------------------------
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=cfg["seed"])
               if is_distributed else None)
    g = torch.Generator().manual_seed(cfg["seed"])
    cpu_budget = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
    req_workers = int(cfg.get("num_workers", 0))
    num_workers = max(0, min(req_workers, cpu_budget))
    if num_workers < req_workers:
        log(f"[data] capping num_workers {req_workers} -> {num_workers} "
            f"(CPUs available: {cpu_budget})")
    # mae_batch_size lets the full pipeline use a big MAE batch without
    # changing the classifier batch.
    batch_size = cfg.get("mae_batch_size", cfg.get("batch_size", 512))
    gpu_data = bool(cfg.get("gpu_data", False)) and is_cuda
    loader = None if gpu_data else DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler, generator=g,
        num_workers=num_workers, pin_memory=pin, collate_fn=_as_batch,
        persistent_workers=(num_workers > 0),
        # Python 3.14 defaults to "forkserver", which pickles the dataset into
        # /dev/shm for the workers: one full copy per rank, which runs out of
        # space at full scale. Forked workers share the parent's memory.
        multiprocessing_context=("fork" if num_workers > 0
                                 and "fork" in mp.get_all_start_methods() else None),
    )

    if gpu_data:
        # The distinct train rows are small enough to live on the GPU, so each
        # batch is gathered there: no DataLoader workers, no host-to-device
        # copies. The shuffle replays exactly what the DataLoader would do, so
        # every batch (and so the training result) is the same.
        rows_d = torch.from_numpy(train_rows.rows).to(device)
        starts_d = torch.from_numpy(train_rows.starts).to(device)
        k_win = train_rows.k
        log(f"[data] train rows on GPU ({rows_d.numel() * 4 / 2**30:.1f} GiB)")

    def epoch_batches(epoch: int):
        if sampler is not None:
            sampler.set_epoch(epoch)       # reshuffle deterministically per epoch
        if not gpu_data:
            for (xb,) in loader:
                yield xb.to(device, non_blocking=pin)
            return
        if sampler is not None:
            order = torch.tensor(list(sampler), dtype=torch.int64)
        else:
            # RandomSampler draws from the DataLoader generator: one seed for
            # the workers when an iterator is created (once with persistent
            # workers, every epoch without), then one permutation per epoch.
            if epoch == 0 or num_workers == 0:
                torch.empty((), dtype=torch.int64).random_(generator=g)
            order = torch.randperm(n, generator=g)
        order = order.to(device)
        for i in range(0, len(order), batch_size):
            j = order[i:i + batch_size]
            yield rows_d[starts_d[j // k_win] + j % k_win]

    mask_frac = cfg.get("mask_frac", 0.4)
    epochs = cfg["mae_epochs"]
    n = len(dataset)
    # Benchmarking: skip a few warmup steps, then time the rest. Only active
    # when a --bench-out path is given, so normal training is unchanged.
    bench = bench_out is not None
    bench_warmup = int(cfg.get("bench_warmup", 10))
    local_batch = batch_size
    step_ms: list[float] = []
    global_step = 0

    net.train()
    for epoch in range(epochs):
        _push(f"mae/epoch_{epoch}")
        epoch_t0 = time.perf_counter()
        # Summed on the device (float64, same arithmetic as a Python float), so
        # logging the loss does not wait for the GPU after every step.
        total = torch.zeros((), dtype=torch.float64, device=device)
        for xb in epoch_batches(epoch):
            # Time this step only after warmup, bracketing with a CUDA sync so
            # async kernels have actually finished before we read the clock.
            timed = bench and global_step >= bench_warmup
            if timed and is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter() if timed else 0.0

            mask = (torch.rand_like(xb) > mask_frac).float()
            opt.zero_grad(set_to_none=True)
            _push("mae/forward")
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=use_amp):
                recon = net(xb * mask)
                loss = loss_fn(recon, xb)
            _pop()
            _push("mae/backward")
            scaler.scale(loss).backward()
            _pop()
            _push("mae/optim")
            scaler.step(opt)
            scaler.update()
            _pop()
            total += loss.detach().double() * len(xb)

            if timed:
                if is_cuda:
                    torch.cuda.synchronize(device)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            global_step += 1
        _pop()
        # Each rank sees ~n/world_size rows; this per-rank mean is a fine
        # progress signal (rank 0 only).
        denom = n / max(world_size, 1)
        log(f"  [MAE] epoch {epoch + 1}/{epochs} recon_mse={total.item() / denom:.4f} "
            f"({time.perf_counter() - epoch_t0:.0f}s)")

    # ---- Benchmark aggregation + JSON (rank 0) -----------------------------
    if bench:
        _write_bench(bench_out, step_ms, local_batch, world_size,
                     is_distributed, device, is_main, log)


    return net.module if is_distributed else net


def train(cfg: dict, bench_out: str | None = None) -> None:
    rank, local_rank, world_size, is_distributed = _dist_info()
    device = _setup(is_distributed, local_rank)
    is_main = rank == 0
    is_cuda = device.type == "cuda"

    def log(*a):
        if is_main:
            print(*a, flush=True)

    # TF32 (Ampere+); harmless elsewhere.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg["seed"])

    # ---- Data --------------------------------------------------------------
    # The MAE trains on every step of every train window. Windows overlap, so
    # it reads them through the per-ROP rows instead of k materialised copies.
    # Default: every rank prepares the same slice. With data_cache set (and
    # DDP), rank 0 prepares once and the other ranks load its train rows.
    cache = None
    if is_distributed and cfg.get("data_cache"):
        tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('MASTER_PORT', '0')}"
        cache = Path(cfg["data_cache"]) / f"mae-train-{tag}.npz"

    prep = None
    if cache is None or is_main:
        log(f"[data] loading telemetry (max_days={cfg['max_days']}, "
            f"max_gnbs={cfg['max_gnbs']}) ...")
        d = load(cfg["dataset_root"], max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])
        labeled = label_frame(
            d.frame,
            load_alarms(str(Path(cfg["dataset_root"]) / "alarms" / "alarms.csv")),
            horizon=cfg["horizon"],
        )
        prep = prepare_rows(labeled, d.feature_cols, k=cfg["window_k"])
        del d, labeled
        train_rows: WindowRows = prep.train
    if cache is not None:
        if is_main:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.name + ".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, rows=train_rows.rows, starts=train_rows.starts)
            os.replace(tmp, cache)
            log(f"[data] shared train rows via {cache}")
        else:
            # Poll instead of a collective: preparing can outlast NCCL timeouts.
            while not cache.exists():
                time.sleep(5)
            with np.load(cache) as z:
                train_rows = WindowRows(z["rows"], z["starts"], cfg["window_k"],
                                        None, None, None)
    in_dim = train_rows.rows.shape[1]
    core = fit_mae_rows(cfg, train_rows, device,
                        (rank, local_rank, world_size, is_distributed, is_cuda),
                        log, bench_out)

    # ---- Checkpoint + eval (rank 0 only) -----------------------------------
    if is_main:
        out = Path(cfg.get("mae_ckpt", "artifacts/mae.pt"))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": core.state_dict(),
             "in_dim": in_dim,
             "latent_dim": cfg["latent_dim"]},
            out,
        )
        log(f"[ckpt] saved MAE -> {out}")

        # Quick reconstruction MSE on the val split as a sanity eval.
        core.eval()
        # Batched, so a large val split doesn't have to fit on the GPU at once.
        val = _WindowStepRows(prep.val.rows, prep.val.starts, prep.val.k)
        sq, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), 262_144):
                idx = np.arange(i, min(i + 262_144, len(val)))
                vflat = val.__getitems__(idx).to(device)
                vrecon = core(vflat)
                sq += float(nn.functional.mse_loss(vrecon, vflat, reduction="sum"))
                count += vflat.numel()
        vmse = sq / max(count, 1)
        log(f"[eval] val recon_mse={vmse:.4f}")

    _cleanup(is_distributed)
    if cache is not None and is_main:
        cache.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the autofm MAE (DDP-capable).")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--dataset-root", default=None,
                    help="Override dataset_root from config (useful on the "
                         "cluster to avoid editing YAML).")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override mae_epochs (handy for short profiling runs).")
    ap.add_argument("--bench-out", default=None,
                    help="If set, measure step time/throughput and write a "
                         "JSON here (rank 0). Training behavior is unchanged.")
    ap.add_argument("--data-cache", default=None,
                    help="DDP only: rank 0 prepares the data once and writes the "
                         "train rows to a file in this directory; the other ranks "
                         "load it instead of preparing the data themselves.")
    ap.add_argument("--gpu-data", action="store_true",
                    help="Keep the train rows on the GPU and gather batches there "
                         "(no DataLoader); same batches and results.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override batch_size (per GPU) from config.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.dataset_root is not None:
        cfg["dataset_root"] = args.dataset_root
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs
    if args.epochs is not None:
        cfg["mae_epochs"] = args.epochs
    if args.data_cache is not None:
        cfg["data_cache"] = args.data_cache
    if args.gpu_data:
        cfg["gpu_data"] = True
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    train(cfg, bench_out=args.bench_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
