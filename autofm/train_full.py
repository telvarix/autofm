"""DDP-capable entry point for the FULL autofm pipeline.

Runs the whole flow - load -> label -> features -> compress (MAE) -> [GAT] ->
classify -> [conformal] -> [advisory] - with the two trainable neural stages
(the MAE compressor and the temporal classifier) scaled across GPUs via
``torchrun``/DistributedDataParallel. The lightweight GAT fusion runs replicated
on each rank; the once-only stages (test eval, conformal, advisory, printing,
metrics JSON) run on **rank 0 only**, after a barrier.

Launch:
    # single process (laptop CPU/MPS, or a single GPU)
    python -m autofm.train_full --config config_local.yaml --use-gat --advisory

    # multi-GPU on one node
    torchrun --nproc_per_node=4 -m autofm.train_full --config config_autofm.yaml

Single-process behavior is intentionally **identical to ``autofm.pipeline``**:
when ``WORLD_SIZE`` is unset or 1 this simply delegates to ``pipeline.run(cfg)``
so there is no divergence and no DDP/NCCL dependency on a laptop. The DDP path
is only taken under ``torchrun`` (``WORLD_SIZE > 1``).

DDP plumbing (env vars, backend override, preflight, bench, rank-0 convention)
is reused from :mod:`autofm.train_mae`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from autofm import pipeline as pl
from autofm.data.features import WindowRows, prepare, prepare_rows
from autofm.data.labels import CLASS_NAMES, label_from_config
from autofm.data.loader import load
from autofm.metrics import compute_all, format_summary, write_prometheus
from autofm.model.classifier import TemporalClassifier, _TemporalEncoder
from autofm.model.compress import MAECompressor, _MAENet, _flatten_time
from autofm.model.conformal import ConformalPredictor
from autofm.model.gat_fusion import GATFusion
from autofm.pipeline import _build_advisory, load_config
from autofm.train_mae import (
    _cleanup,
    _dist_info,
    _setup,
    _write_bench,
    fit_mae_rows,
)
from sklearn.metrics import classification_report


# --------------------------------------------------------------------------- #
# DDP training of one nn.Module over a (features[, labels]) tensor dataset
# --------------------------------------------------------------------------- #
def _ddp_wrap(model: nn.Module, is_distributed: bool, local_rank: int, is_cuda: bool):
    if is_distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if is_cuda else None,
            output_device=local_rank if is_cuda else None,
        )
    return model


def _make_loader(dataset, cfg, rank, world_size, is_distributed, pin):
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True, seed=cfg["seed"])
               if is_distributed else None)
    g = torch.Generator().manual_seed(cfg["seed"])
    return sampler, DataLoader(
        dataset, batch_size=cfg.get("batch_size", 512),
        shuffle=(sampler is None), sampler=sampler, generator=g,
        num_workers=cfg.get("num_workers", 0), pin_memory=pin,
        persistent_workers=(cfg.get("num_workers", 0) > 0),
    )


def _train_mae_ddp(cfg, prep, device, ctx, bench_out, log) -> _MAENet:
    """DDP-train the MAE on the train split; return the trained core _MAENet."""
    rank, local_rank, world_size, is_distributed, is_cuda = ctx
    flat = _flatten_time(prep.train.X).astype(np.float32, copy=False)
    core = _MAENet(flat.shape[1], cfg["latent_dim"]).to(device)
    net = _ddp_wrap(core, is_distributed, local_rank, is_cuda)
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

    sampler, loader = _make_loader(
        TensorDataset(torch.from_numpy(flat)), cfg, rank, world_size,
        is_distributed, pin)

    mask_frac = cfg.get("mask_frac", 0.4)
    epochs = cfg["mae_epochs"]
    n = len(flat)
    bench = bench_out is not None
    bench_warmup = int(cfg.get("bench_warmup", 10))
    step_ms: list[float] = []
    gstep = 0

    net.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        _push(f"mae/epoch_{epoch}")
        total = 0.0
        for (xb,) in loader:
            timed = bench and gstep >= bench_warmup
            if timed and is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter() if timed else 0.0

            xb = xb.to(device, non_blocking=pin)
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
            total += loss.item() * len(xb)
            if timed:
                if is_cuda:
                    torch.cuda.synchronize(device)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            gstep += 1
        _pop()
        denom = n / max(world_size, 1)
        log(f"  [MAE] epoch {epoch + 1}/{epochs} recon_mse={total / denom:.4f}")

    if bench:
        _write_bench(bench_out, step_ms, cfg.get("batch_size", 512), world_size,
                     is_distributed, device, rank == 0, log)
    return net.module if is_distributed else net


def _train_clf_ddp(cfg, Ztr, ytr, device, ctx, log) -> _TemporalEncoder:
    """DDP-train the temporal classifier's encoder; return the trained core."""
    from autofm.model.evidential import edl_mse_loss

    rank, local_rank, world_size, is_distributed, is_cuda = ctx
    head = cfg.get("clf_head", "softmax")
    n_classes = len(CLASS_NAMES)
    core = _TemporalEncoder(
        cfg["latent_dim"], n_classes,
        head=head, backbone=cfg.get("clf_backbone", "transformer"),
        window_k=cfg["window_k"], patch_size=cfg.get("patch_size", 5),
    ).to(device)
    net = _ddp_wrap(core, is_distributed, local_rank, is_cuda)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Inverse-frequency class weights (same as TemporalClassifier), raised to
    # class_weight_power: 1 = full inverse frequency (default), 0.5 = square
    # root (gentler), 0 = unweighted.
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    power = float(cfg.get("class_weight_power", 1.0))
    cw = torch.tensor((counts.sum() / (n_classes * counts)) ** power,
                      dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=cw)

    pin = is_cuda
    sampler, loader = _make_loader(
        TensorDataset(torch.from_numpy(Ztr.astype(np.float32, copy=False)),
                      torch.from_numpy(ytr.astype(np.int64))),
        cfg, rank, world_size, is_distributed, pin)

    epochs = cfg["clf_epochs"]
    anneal_epochs = int(cfg.get("anneal_epochs", 10))
    net.train()
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        anneal = min(1.0, epoch / max(anneal_epochs, 1))
        total = 0.0
        seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=pin), yb.to(device, non_blocking=pin)
            out = net(xb)
            if head == "evidential":
                loss = edl_mse_loss(out, yb, n_classes=n_classes,
                                    anneal=anneal, class_weight=cw)
            else:
                loss = ce(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        log(f"  [clf:{head}] epoch {epoch:02d}/{epochs} "
            f"loss={total / max(seen, 1):.4f}")
    return net.module if is_distributed else net


def _with_diff(Z: np.ndarray) -> np.ndarray:
    """Append the step-to-step change of every input, ``(N, k, L) -> (N, k, 2L)``
    (zero change at the first step of each window)."""
    d = np.zeros_like(Z)
    np.subtract(Z[:, 1:], Z[:, :-1], out=d[:, 1:])
    return np.concatenate([Z, d], axis=2)


def _compress_rows(comp, w: WindowRows) -> np.ndarray:
    """Latent windows ``(N, k, L)`` for one split, without materialising the
    ``(N, k, F)`` input windows.

    Both compressors map each ROP row on its own, so encoding every distinct
    row once and gathering by window gives the same latents as
    ``comp.transform(w.windows())`` at a fraction of the memory and compute.
    """
    z_rows = comp.transform(w.rows[:, None, :])[:, 0, :]
    return z_rows[w.row_index()]


# --------------------------------------------------------------------------- #
# Rank-0 once-only tail: compress transforms, GAT, eval, conformal, advisory
# --------------------------------------------------------------------------- #
def _finish_rank0(cfg, prep, comp, clf_core, t0, log, gat=None, Z=None) -> dict:
    """Run the deterministic tail on rank 0 and print/return results.

    ``comp`` is the already-fitted compressor (MAE wrapper or PCA); ``clf_core``
    is the DDP-trained classifier encoder. ``gat`` is the SAME (seed-fixed)
    GATFusion instance used to enrich the train split during training, reused
    here so eval sees an identical transform (no second random init).
    ``Z`` optionally carries the already-compressed ``(train, val, test)``
    latents (low-memory mode), so the splits are not compressed again.
    """
    # Compress all splits with the already-trained compressor.
    if Z is None:
        Z = (comp.transform(prep.train.X), comp.transform(prep.val.X),
             comp.transform(prep.test.X))
    Ztr, Zval, Zte = Z

    # Optional GAT fusion - reuse the SAME instance that enriched the train
    # split during training (seed-fixed, untrained weights), so train and eval
    # share one transform. Grouping uses the real cell -> gNB topology map.
    if gat is not None:
        log(f"[gat] topology fusion active "
            f"({'PyG' if gat.gat.use_pyg else 'dense-fallback'}) - "
            f"enriching latents with within-gNB neighbour context")
        Ztr = gat.transform(Ztr, prep.train.cells)
        Zval = gat.transform(Zval, prep.val.cells)
        Zte = gat.transform(Zte, prep.test.cells)

    # Wrap the DDP-trained classifier core in a TemporalClassifier for its
    # predict_proba / uncertainty helpers (no re-training).
    head = cfg.get("clf_head", "softmax")
    clf = TemporalClassifier(
        latent_dim=cfg["latent_dim"], n_classes=len(CLASS_NAMES),
        epochs=cfg["clf_epochs"], seed=cfg["seed"], head=head,
        backbone=cfg.get("clf_backbone", "transformer"),
        window_k=cfg["window_k"], patch_size=cfg.get("patch_size", 5),
    )
    clf._net = clf_core

    proba = clf.predict_proba(Zte)
    _, unc = clf.predict_with_uncertainty(Zte)
    pred = proba.argmax(axis=1) if len(proba) else np.empty(0, dtype=np.int64)
    mean_unc = float(unc.mean()) if len(unc) else float("nan")

    causal_chain: list[str] = []
    if gat is not None and len(pred):
        fault_rows = np.where(pred != 0)[0]
        if len(fault_rows):
            causal_chain = gat.causal_chain(
                str(prep.test.cells[int(fault_rows[0])]), depth=3)

    metrics = compute_all(prep.test.y_class, pred, CLASS_NAMES,
                          proba=proba, uncertainty=unc)
    test_f1 = metrics["multiclass"]["macro_f1"]

    # Conformal (val slice as calibration), same as pipeline.
    conformal_coverage = conformal_set_size = None
    if cfg.get("use_conformal", False):
        proba_val = clf.predict_proba(Zval)
        if len(proba_val) and len(proba):
            cp = ConformalPredictor(confidence_level=0.95)
            cp.calibrate(proba_val, prep.val.y_class)
            sets = cp.predict_set(proba)
            conformal_coverage = cp.empirical_coverage(sets, prep.test.y_class)
            conformal_set_size = cp.mean_set_size(sets)
            metrics["conformal"] = {
                "target_coverage": 0.95,
                "empirical_coverage": conformal_coverage,
                "mean_set_size": conformal_set_size,
            }

    present = sorted(set(prep.test.y_class) | set(pred))
    report = classification_report(
        prep.test.y_class, pred, labels=present,
        target_names=[CLASS_NAMES[i] for i in present], zero_division=0)
    elapsed = time.perf_counter() - t0

    unc_label = ("mean uncertainty" if head == "evidential"
                 else "mean confidence-gap")
    log("\n" + "=" * 64)
    log(f"autofm FM pipeline (DDP) - TEST macro-F1 = {test_f1:.4f}  "
        f"(compressor={cfg['compressor']}, "
        f"backbone={cfg.get('clf_backbone')}, head={head}, "
        f"horizon={cfg['horizon']})")
    log(f"{unc_label}: {mean_unc:.4f}")
    log("=" * 64)
    log(format_summary(metrics, CLASS_NAMES))
    log("\nPer-class report:")
    log(report)
    if gat is not None:
        log(f"[gat] causal chain (predicted fault): "
            f"{' -> '.join(causal_chain) if causal_chain else '(none)'}")
    if conformal_coverage is not None:
        log(f"[conformal] target 95% coverage | "
            f"empirical {conformal_coverage * 100:.1f}% | "
            f"mean set size {conformal_set_size:.2f}")

    advisory_fields = None
    if cfg.get("advisory", False):
        advisory_fields = _build_advisory(cfg, proba, pred, unc,
                                          prep.test.cells, gat)

    # Metrics JSON (rank 0 only).
    metrics_out = cfg.get("metrics_out")
    if metrics_out:
        payload = dict(metrics)
        payload["conformal_coverage"] = conformal_coverage
        payload["conformal_set_size"] = conformal_set_size
        if advisory_fields is not None:
            payload["advisory"] = advisory_fields
        payload["config"] = {k: cfg[k] for k in (
            "compressor", "clf_backbone", "clf_head", "horizon", "window_k",
            "latent_dim", "max_days", "max_gnbs") if k in cfg}
        payload["elapsed_sec"] = round(elapsed, 1)
        payload["world_size"] = _dist_info()[2]
        out_path = Path(metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        log(f"[metrics] wrote {out_path}")

    return {"test_macro_f1": test_f1, "advisory": advisory_fields}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train_full(cfg: dict, bench_out: str | None = None) -> dict | None:
    rank, local_rank, world_size, is_distributed = _dist_info()

    # Single-process path == pipeline.run, guaranteeing identical laptop output.
    # Low-memory mode takes the path below even for one process: pipeline.run
    # materialises every window.
    low_mem = bool(cfg.get("low_mem", False))
    if not is_distributed and not low_mem:
        if bench_out:
            # Still support --bench-out single-process by timing the MAE below;
            # but for identical-output guarantee we run the standard pipeline.
            print("[train_full] single process: delegating to pipeline.run "
                  "(identical to `python -m autofm.pipeline`)", flush=True)
        return pl.run(cfg)

    # ---- Distributed path --------------------------------------------------
    device = _setup(is_distributed, local_rank)
    is_cuda = device.type == "cuda"
    is_main = rank == 0
    ctx = (rank, local_rank, world_size, is_distributed, is_cuda)

    def log(*a):
        if is_main:
            print(*a, flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg["seed"])
    t0 = time.perf_counter()

    # Data (every rank loads the same slice; sharding happens in the samplers).
    log(f"[data] loading telemetry (max_days={cfg['max_days']}, "
        f"max_gnbs={cfg['max_gnbs']}) ...")
    d = load(cfg["dataset_root"], max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])
    labeled = label_from_config(d.frame, cfg)
    log(f"[labels] mode={cfg.get('label_mode', 'forecast')} "
        f"scopes={cfg.get('alarm_scopes') or 'all'} "
        f"positive rate {labeled['will_fault'].mean():.4f}")
    # Low-memory mode keeps each split as per-ROP rows plus window offsets
    # instead of k overlapping copies of every row (about k times smaller).
    prep = (prepare_rows if low_mem else prepare)(labeled, d.feature_cols,
                                                  k=cfg["window_k"])
    # Authoritative cell -> gNB topology map (loader fdn parse) for the GAT.
    cell_to_gnb = (
        d.frame[["cell", "gnb"]].dropna().astype(str)
        .drop_duplicates("cell").set_index("cell")["gnb"].to_dict())
    del d, labeled
    if low_mem:
        log(f"[data] low-mem: train {len(prep.train.starts)} windows "
            f"({len(prep.train.rows)} rows)  val {len(prep.val.starts)}  "
            f"test {len(prep.test.starts)}  k={cfg['window_k']}")
    else:
        log(f"[data] train {prep.train.X.shape}  val {prep.val.X.shape}  "
            f"test {prep.test.X.shape}")

    # Stage 1: compressor. MAE trains under DDP; PCA (sklearn, non-neural) is
    # fit replicated on each rank - cheap and deterministic.
    compressor_kind = cfg.get("compressor", "mae")
    if compressor_kind == "mae":
        log(f"[1/2] MAE ({cfg['mae_epochs']} epochs, DDP x{world_size}) ...")
        mae_core = (fit_mae_rows(cfg, prep.train, device, ctx, log, bench_out)
                    if low_mem else
                    _train_mae_ddp(cfg, prep, device, ctx, bench_out, log))
        comp = MAECompressor(latent_dim=cfg["latent_dim"], seed=cfg["seed"])
        comp._net = mae_core
    else:
        log(f"[1/2] PCA compressor (replicated, non-DDP) ...")
        from autofm.model.compress import build_compressor
        comp = build_compressor("pca", latent_dim=cfg["latent_dim"],
                                seed=cfg["seed"])
        comp.fit(prep.train.windows() if low_mem else prep.train.X)
        mae_core = None  # no neural MAE core in the PCA path

    # Compress the TRAIN split (replicated) to feed the classifier.
    Z = None
    if low_mem:
        Ztr = _compress_rows(comp, prep.train)
        if is_main:  # only rank 0 evaluates
            Z = (Ztr, _compress_rows(comp, prep.val), _compress_rows(comp, prep.test))
        log(f"[compress] latents train {Ztr.shape} "
            f"({Ztr.nbytes / 2**30:.1f} GiB)")
    else:
        Ztr = comp.transform(prep.train.X)
    if cfg.get("add_diff"):
        if not low_mem:
            raise ValueError("add_diff needs --low-mem")
        Ztr = _with_diff(Ztr)
        if Z is not None:
            Z = (Ztr, _with_diff(Z[1]), _with_diff(Z[2]))
        cfg["latent_dim"] = Ztr.shape[2]
        log(f"[compress] + step-to-step change: inputs {Ztr.shape} "
            f"({Ztr.nbytes / 2**30:.1f} GiB)")

    # Optional GAT enrich of the train latents before classifier training.
    # ONE seed-fixed instance, reused for eval in _finish_rank0 (identical
    # weights across train and eval).
    gat = None
    if cfg.get("use_gat", False):
        gat = GATFusion(latent_dim=cfg["latent_dim"], seed=cfg["seed"],
                        cell_to_gnb=cell_to_gnb)
        Ztr = gat.fit_transform(Ztr, prep.train.cells)

    # Stage 2: DDP-train the temporal classifier.
    log(f"[2/2] Classifier ({cfg['clf_epochs']} epochs, DDP x{world_size}) ...")
    clf_core = _train_clf_ddp(cfg, Ztr, prep.train.y_class, device, ctx, log)

    # Barrier: ensure all ranks finished training before rank-0 eval.
    if is_distributed:
        torch.distributed.barrier()

    result = None
    if is_main:
        result = _finish_rank0(cfg, prep, comp, clf_core, t0, log, gat=gat, Z=Z)
        log(f"Total wall time: {time.perf_counter() - t0:.1f}s on {device}")

    _cleanup(is_distributed)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the full autofm pipeline (DDP-capable).")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--compressor", choices=["pca", "mae"], default=None)
    ap.add_argument("--backbone", choices=["transformer", "patchtst"], default=None)
    ap.add_argument("--head", choices=["softmax", "evidential"], default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    ap.add_argument("--use-gat", action="store_true", default=False)
    ap.add_argument("--use-conformal", action="store_true", default=False)
    ap.add_argument("--advisory", action="store_true", default=False)
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--bench-out", default=None,
                    help="Record MAE throughput JSON (rank 0) for scaling study.")
    ap.add_argument("--low-mem", action="store_true",
                    help="Keep splits as per-ROP rows and compress each row once "
                         "(needed for long histories; also with one GPU).")
    ap.add_argument("--gpu-data", action="store_true",
                    help="Low-mem MAE: keep train rows on the GPU (no DataLoader).")
    ap.add_argument("--mae-batch-size", type=int, default=None,
                    help="Low-mem MAE batch (the classifier keeps batch_size).")
    ap.add_argument("--mae-lr", type=float, default=None)
    ap.add_argument("--label-mode", choices=["forecast", "active"], default=None,
                    help="forecast: an alarm starts within the horizon (default); "
                         "active: an alarm is active now (detection).")
    ap.add_argument("--alarm-scopes", default=None,
                    help="Comma-separated subset of cell,gnb,region: only these "
                         "alarms make labels (default: all).")
    ap.add_argument("--clf-epochs", type=int, default=None)
    ap.add_argument("--class-weight-power", type=float, default=None,
                    help="Classifier class weights = inverse frequency ** power "
                         "(1 default, 0.5 gentler, 0 unweighted).")
    ap.add_argument("--mae-epochs", type=int, default=None)
    ap.add_argument("--add-diff", action="store_true",
                    help="Low-mem: also feed each input's change since the "
                         "previous ROP.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.compressor:
        cfg["compressor"] = args.compressor
    if args.backbone:
        cfg["clf_backbone"] = args.backbone
    if args.head:
        cfg["clf_head"] = args.head
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs
    if args.use_gat:
        cfg["use_gat"] = True
    if args.use_conformal:
        cfg["use_conformal"] = True
    if args.advisory:
        cfg["advisory"] = True
    if args.metrics_out is not None:
        cfg["metrics_out"] = args.metrics_out
    if args.low_mem:
        cfg["low_mem"] = True
    if args.gpu_data:
        cfg["gpu_data"] = True
    if args.mae_batch_size is not None:
        cfg["mae_batch_size"] = args.mae_batch_size
    if args.mae_lr is not None:
        cfg["mae_lr"] = args.mae_lr
    if args.label_mode is not None:
        cfg["label_mode"] = args.label_mode
    if args.alarm_scopes is not None:
        cfg["alarm_scopes"] = [s.strip() for s in args.alarm_scopes.split(",") if s.strip()]
    if args.clf_epochs is not None:
        cfg["clf_epochs"] = args.clf_epochs
    if args.class_weight_power is not None:
        cfg["class_weight_power"] = args.class_weight_power
    if args.add_diff:
        cfg["add_diff"] = True
    if args.mae_epochs is not None:
        cfg["mae_epochs"] = args.mae_epochs

    train_full(cfg, bench_out=args.bench_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
