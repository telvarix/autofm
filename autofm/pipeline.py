"""End-to-end autofm FM pipeline (first runnable slice).

Wires the stages into one runnable path:

    load multi-level telemetry
      -> label from the alarm feed (forecast horizon)
      -> clean / split / scale / window
      -> compress each ROP to a latent (PCA or MAE)
      -> temporal classifier (PatchTST-lite) over the window
      -> report macro-F1 on the held-out test split

This is the baseline chain the richer autofm components (GAT, evidential heads,
conformal calibration) will slot into later. It runs at smoke scale on CPU/MPS
and scales up by widening the data knobs in the config.

Run:
    python -m autofm.pipeline --config config_autofm.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report

from autofm.data.features import prepare
from autofm.data.labels import CLASS_NAMES, label_from_config
from autofm.data.loader import load
from autofm.metrics import compute_all, format_summary, write_prometheus
from autofm.model.classifier import TemporalClassifier
from autofm.advisory.bus import AnalysisBus, Message
from autofm.advisory.cm_stub import CMStub
from autofm.advisory.correlation import CorrelationEngine
from autofm.advisory.pm_stub import PMStub
from autofm.model.compress import build_compressor
from autofm.model.conformal import ConformalPredictor
from autofm.model.gat_fusion import GATFusion, gnb_of_cell

DEFAULT_CONFIG = {
    "dataset_root": "./data/autofm-dataset-ie",
    "max_days": 14,
    "max_gnbs": 40,
    "horizon": 5,
    "window_k": 32,
    "latent_dim": 32,
    "compressor": "mae",          # "pca" | "mae"
    "mae_epochs": 6,
    "amp": True,                  # mixed precision (CUDA only; ignored on CPU/MPS)
    "num_workers": 8,             # DataLoader workers for MAE training
    "clf_epochs": 12,
    "clf_head": "softmax",        # "softmax" (baseline) | "evidential" (+uncertainty)
    "clf_backbone": "transformer",# "transformer" | "patchtst"
    "patch_size": 5,              # patch length for the patchtst backbone
    "use_gat": False,             # insert GAT topology fusion between compress + classify
    "use_conformal": False,       # calibrate 95% conformal coverage sets on the val slice
    "advisory": False,            # build an FM/PM/CM advisory from the model's real prediction
    "seed": 42,
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and Path(path).exists():
        with open(path) as fh:
            cfg.update(yaml.safe_load(fh) or {})
    return cfg


@contextlib.contextmanager
def nvtx_stage(name: str):
    """Push/pop an NVTX range for a pipeline stage, CUDA-only.

    Makes each stage show up as a named span on the Nsight Systems timeline
    (outer ranges around the MAE's existing inner mae/* ranges). A pure no-op
    when CUDA is absent, so laptop CPU/MPS runs are unchanged.
    """
    on = torch.cuda.is_available()
    if on:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if on:
            torch.cuda.nvtx.range_pop()


def _build_advisory(cfg, proba, pred, unc, cells, gat) -> dict | None:
    """Build + print an FM/PM/CM advisory from the model's real prediction.

    Picks one representative example: the highest-probability NON-Normal
    prediction across the test set. Returns the rendered advisory fields (for
    the metrics JSON), or ``None`` if nothing non-Normal was predicted.
    """
    # Class 0 is Normal; faults are classes 1..K-1.
    if len(proba) == 0 or proba.shape[1] <= 1:
        print("[advisory] no fault predicted in test window; skipping advisory")
        return None
    fault_rows = np.where(pred != 0)[0]
    if len(fault_rows) == 0:
        print("[advisory] no fault predicted in test window; skipping advisory")
        return None

    # Among predicted faults, pick the one the model is most confident about:
    # the max probability mass on its (non-Normal) predicted class.
    conf_of_pred = proba[fault_rows, pred[fault_rows]]
    ex = int(fault_rows[int(np.argmax(conf_of_pred))])

    cell = str(cells[ex])
    # Prefer the authoritative cell -> gNB topology map (from the loader's fdn
    # parse, carried on the GAT); fall back to the lossy regex only if absent.
    if gat is not None and getattr(gat, "cell_to_gnb", None):
        gnb = str(gat.cell_to_gnb.get(cell, gnb_of_cell(cell)))
    else:
        gnb = gnb_of_cell(cell)
    fault_class = CLASS_NAMES[int(pred[ex])]
    # confidence = 1 - uncertainty for this example, clipped to [0, 1].
    conf = 1.0 - float(unc[ex]) if ex < len(unc) else float(proba[ex, pred[ex]])
    conf = float(np.clip(conf, 0.0, 1.0))

    # Origin / causal chain: GAT chain of cell ids when available, else the cell.
    if gat is not None:
        chain = gat.causal_chain(cell, depth=3)
        origin = " -> ".join(chain) if chain else cell
    else:
        origin = cell

    fm = Message(
        pillar="FM", kind="fault_prediction",
        payload={
            "fault_class": fault_class,
            "probable_cause": f"model-predicted {fault_class}",
            "gnb": gnb,
            "origin_node": origin,
            "affected_cell": cell,
            "horizon_rops": cfg["horizon"],
            "confidence": conf,
        },
    )

    bus = AnalysisBus()
    bus.publish(fm)
    PMStub(bus).assess()
    CMStub(bus).propose()
    advisory = CorrelationEngine(bus).correlate()
    if advisory is None:
        print("[advisory] correlation produced no advisory; skipping")
        return None

    print("\n=== AutoFM advisory (from model prediction) ===")
    print(advisory.render())
    return {
        "what": advisory.what,
        "where": advisory.where,
        "when": advisory.when,
        "why": advisory.why,
        "action": advisory.action,
        "confidence": advisory.confidence,
    }


def run(cfg: dict) -> dict:
    t0 = time.perf_counter()
    root = cfg["dataset_root"]

    print(f"[1/5] Loading telemetry (max_days={cfg['max_days']}, "
          f"max_gnbs={cfg['max_gnbs']}) ...")
    with nvtx_stage("stage/1_load"):
        d = load(root, max_days=cfg["max_days"], max_gnbs=cfg["max_gnbs"])
    print(f"      {len(d.frame)} rows, {len(d.feature_cols)} raw features, "
          f"{d.frame['cell'].nunique()} cells, {len(d.timestamps)} ROPs")

    print(f"[2/5] Labeling from alarms (horizon={cfg['horizon']} ROPs) ...")
    with nvtx_stage("stage/2_label"):
        labeled = label_from_config(d.frame, cfg)
    print(f"      positive rate {labeled['will_fault'].mean():.3f}")

    # Authoritative cell -> gNB topology map from the loader's fdn parse. Used
    # by the GAT (correct within-gNB grouping) and the advisory (real gNB).
    cell_to_gnb = (
        d.frame[["cell", "gnb"]].dropna().astype(str)
        .drop_duplicates("cell").set_index("cell")["gnb"].to_dict())

    print(f"[3/5] Windowing (k={cfg['window_k']}) + leakage-safe split ...")
    with nvtx_stage("stage/3_features"):
        prep = prepare(labeled, d.feature_cols, k=cfg["window_k"])
    print(f"      train {prep.train.X.shape}  val {prep.val.X.shape}  "
          f"test {prep.test.X.shape}  ({len(prep.feature_cols)} features)")

    print(f"[4/5] Compressing with '{cfg['compressor']}' "
          f"-> latent {cfg['latent_dim']} ...")
    comp_kw = {"seed": cfg["seed"]}
    if cfg["compressor"] == "mae":
        comp_kw["epochs"] = cfg["mae_epochs"]
        comp_kw["amp"] = cfg.get("amp", True)
        comp_kw["num_workers"] = cfg.get("num_workers", 0)
    comp = build_compressor(cfg["compressor"], latent_dim=cfg["latent_dim"], **comp_kw)
    with nvtx_stage("stage/4_compress"):
        comp.fit(prep.train.X)
        Ztr = comp.transform(prep.train.X)
        Zval = comp.transform(prep.val.X)
        Zte = comp.transform(prep.test.X)
    if cfg["compressor"] == "pca":
        print(f"      PCA explained variance: {comp.explained_variance:.3f}")

    # Optional GAT topology-fusion stage. No-op unless use_gat is set. It
    # enriches each window's latent with neighbour-cell context and preserves
    # the (N, k, latent_dim) contract, so the classifier is unchanged.
    gat = None
    if cfg.get("use_gat", False):
        gat = GATFusion(latent_dim=cfg["latent_dim"], seed=cfg["seed"],
                        cell_to_gnb=cell_to_gnb)
        print(f"[gat] topology fusion active "
              f"({'PyG' if gat.gat.use_pyg else 'dense-fallback'}) - "
              f"enriching latents with within-gNB neighbour context")
        with nvtx_stage("stage/5_gat"):
            Ztr = gat.fit_transform(Ztr, prep.train.cells)
            Zval = gat.transform(Zval, prep.val.cells)
            Zte = gat.transform(Zte, prep.test.cells)

    head = cfg.get("clf_head", "softmax")
    backbone = cfg.get("clf_backbone", "transformer")
    print(f"[5/5] Training temporal classifier ({cfg['clf_epochs']} epochs, "
          f"backbone={backbone}, head={head}) ...")
    clf = TemporalClassifier(
        latent_dim=cfg["latent_dim"], n_classes=len(CLASS_NAMES),
        epochs=cfg["clf_epochs"], seed=cfg["seed"], head=head,
        backbone=backbone, window_k=cfg["window_k"],
        patch_size=cfg.get("patch_size", 5),
    )
    with nvtx_stage("stage/6_classify"):
        clf.fit(Ztr, prep.train.y_class, Zval, prep.val.y_class)

    # Final test metrics. One forward pass: get probabilities, derive the class
    # prediction and uncertainty from them, so pred / proba / metrics are all
    # consistent and we don't re-run the net three times.
    proba = clf.predict_proba(Zte)
    _, unc = clf.predict_with_uncertainty(Zte)
    pred = (proba.argmax(axis=1) if len(proba)
            else np.empty(0, dtype=np.int64))
    mean_unc = float(unc.mean()) if len(unc) else float("nan")

    # If GAT is active, trace a causal chain for one predicted-fault example
    # (first test window whose predicted class is a non-Normal fault). This is
    # the "why" that later feeds the advisory layer.
    causal_chain: list[str] = []
    if gat is not None and len(pred):
        fault_rows = np.where(pred != 0)[0]
        if len(fault_rows):
            ex = int(fault_rows[0])
            causal_chain = gat.causal_chain(str(prep.test.cells[ex]), depth=3)

    metrics = compute_all(
        prep.test.y_class, pred, CLASS_NAMES, proba=proba, uncertainty=unc,
    )
    test_f1 = metrics["multiclass"]["macro_f1"]

    # Optional conformal prediction. No-op unless use_conformal is set. The
    # validation split is the calibration slice (the model never trained on it),
    # so coverage is honest. Computes 95% sets on the test split.
    conformal_coverage = None
    conformal_set_size = None
    if cfg.get("use_conformal", False):
        with nvtx_stage("stage/7_conformal"):
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
            else:
                print("[conformal] skipped - empty validation or test split")
    present = sorted(set(prep.test.y_class) | set(pred))
    report = classification_report(
        prep.test.y_class, pred,
        labels=present, target_names=[CLASS_NAMES[i] for i in present],
        zero_division=0,
    )
    elapsed = time.perf_counter() - t0

    unc_label = ("mean uncertainty" if head == "evidential"
                 else "mean confidence-gap")
    bf = metrics["binary_fault"]
    print("\n" + "=" * 64)
    print(f"autofm FM pipeline - TEST macro-F1 = {test_f1:.4f}  "
          f"(compressor={cfg['compressor']}, backbone={backbone}, "
          f"head={head}, horizon={cfg['horizon']})")
    print(f"{unc_label}: {mean_unc:.4f}")
    print("=" * 64)
    print(format_summary(metrics, CLASS_NAMES))
    print("\nPer-class report:")
    print(report)
    if gat is not None:
        if causal_chain:
            print(f"[gat] causal chain (predicted fault): "
                  f"{' -> '.join(causal_chain)}")
        else:
            print("[gat] causal chain: no non-Normal prediction to trace")
    if conformal_coverage is not None:
        print(f"[conformal] target 95% coverage | "
              f"empirical {conformal_coverage * 100:.1f}% | "
              f"mean set size {conformal_set_size:.2f}")
    print(f"Total wall time: {elapsed:.1f}s on device "
          f"{getattr(clf, 'device', 'cpu')}")

    # Optional advisory built from the model's REAL prediction (not hardcoded).
    advisory_fields = None
    if cfg.get("advisory", False):
        with nvtx_stage("stage/8_advisory"):
            advisory_fields = _build_advisory(
                cfg, proba, pred, unc, prep.test.cells, gat)

    # Optional machine-readable metrics dump for tracking across runs.
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
        out_path = Path(metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[metrics] wrote {out_path}")

    # Optional Prometheus exposition file (node_exporter textfile collector).
    prom_out = cfg.get("prom_out")
    if prom_out:
        prom_labels = {
            "compressor": str(cfg.get("compressor", "")),
            "backbone": str(cfg.get("clf_backbone", "")),
            "head": str(head),
            "horizon": str(cfg.get("horizon", "")),
        }
        write_prometheus(prom_out, metrics, prom_labels)
        print(f"[metrics] wrote Prometheus exposition -> {prom_out}")

    result = {"test_macro_f1": test_f1, "mean_uncertainty": mean_unc,
              "elapsed_sec": elapsed, "use_gat": gat is not None,
              "causal_chain": causal_chain,
              "conformal_coverage": conformal_coverage,
              "conformal_set_size": conformal_set_size,
              "advisory": advisory_fields}
    result.update({
        "weighted_f1": metrics["multiclass"]["weighted_f1"],
        "balanced_accuracy": metrics["multiclass"]["balanced_accuracy"],
        "fault_recall": bf["recall"],
        "fault_precision": bf["precision"],
        "fault_pr_auc": bf.get("pr_auc"),
        "fault_roc_auc": bf.get("roc_auc"),
        "metrics": metrics,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the autofm FM pipeline.")
    ap.add_argument("--config", default="config_autofm.yaml")
    ap.add_argument("--compressor", choices=["pca", "mae"], default=None,
                    help="Override the compressor from config.")
    ap.add_argument("--head", choices=["softmax", "evidential"], default=None,
                    help="Override the classifier head from config.")
    ap.add_argument("--backbone", choices=["transformer", "patchtst"], default=None,
                    help="Override the temporal backbone from config.")
    ap.add_argument("--use-gat", action="store_true", default=False,
                    help="Insert the GAT topology-fusion stage (default off).")
    ap.add_argument("--use-conformal", action="store_true", default=False,
                    help="Calibrate 95%% conformal coverage sets on the val slice.")
    ap.add_argument("--advisory", action="store_true", default=False,
                    help="Build an FM/PM/CM advisory from the model's prediction.")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--max-gnbs", type=int, default=None)
    ap.add_argument("--metrics-out", default=None,
                    help="Write the full metrics bundle as JSON to this path.")
    ap.add_argument("--prom-out", default=None,
                    help="Write Prometheus exposition metrics (.prom) to this "
                         "path for the node_exporter textfile collector.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.compressor:
        cfg["compressor"] = args.compressor
    if args.head:
        cfg["clf_head"] = args.head
    if args.backbone:
        cfg["clf_backbone"] = args.backbone
    if args.use_gat:
        cfg["use_gat"] = True
    if args.use_conformal:
        cfg["use_conformal"] = True
    if args.advisory:
        cfg["advisory"] = True
    if args.max_days is not None:
        cfg["max_days"] = args.max_days
    if args.max_gnbs is not None:
        cfg["max_gnbs"] = args.max_gnbs
    if args.metrics_out is not None:
        cfg["metrics_out"] = args.metrics_out
    if args.prom_out is not None:
        cfg["prom_out"] = args.prom_out

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
