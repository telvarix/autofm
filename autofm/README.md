# autofm FM pipeline

End-to-end fault-prediction pipeline for multi-level O-RAN telemetry, labeled
from an X.733 alarm feed. This is the FM track described in
[`../docs/SYSTEM_DESIGN.md`](../docs/SYSTEM_DESIGN.md); the richer components
(GAT, evidential heads, conformal calibration) slot into the same chain.

## The chain

```
load multi-level telemetry      autofm/data/loader.py
  -> label from alarm feed       autofm/data/labels.py     (forecast horizon)
  -> clean / split / window      autofm/data/features.py   (leakage-safe)
  -> compress each ROP           autofm/model/compress.py  (PCA or MAE -> latent)
  -> temporal classifier         autofm/model/classifier.py (PatchTST-lite)
  -> fault-type macro-F1         autofm/pipeline.py
```

## Dataset layout it expects

```
autofm-dataset-ie/
├── 2025/<MM>/<level>/<YYYY-MM-DD>/        # 15-min ROP telemetry, ;-delimited
│   ├── infra/      OCloudNode_infra_<date>.csv          host metrics
│   ├── platform/   ManagedNFService_platform_<date>.csv container metrics
│   ├── pm/         <group>_pm_<date>.csv                RAN performance (5 groups)
│   └── cm/         <object>_cm_<date>.csv               RAN config (34 groups)
└── alarms/alarms.csv                                    X.733 alarm feed (labels)
```

Run `python -m autofm.profile_dataset --root <path>` to regenerate
[`../docs/DATA_NOTES.md`](../docs/DATA_NOTES.md), the auto-profiled summary of
levels, groups, cadence, entities, and the alarm taxonomy.

### Faster loading: convert to Parquet once

Parsing the CSVs dominates every run. Convert the groups the loader reads to
typed Parquet (needs `pyarrow`), then point `dataset_root` at the output; the
loader prefers Parquet and falls back to CSV:

```bash
python -m autofm.data.to_parquet --src <csv dataset> --dst <parquet dataset> --workers 8
```

Same frame as from the CSVs; on 14 days x 20 gNBs loading drops from 145 s to
12 s, and the four groups for all 182 days take 3.2 GB instead of about 53 GB.
Memory use is unchanged: the loader still holds every loaded day in RAM.

## How labeling works

Telemetry carries no inline label. Each alarm is an interval
`[eventTime, clearTime]` on an `fdn`. The labeler maps an alarm to the cells it
affects (direct `NRCellDU`; or all cells of a gNB for `S1ULink` /
`ManagedNFService`; or all cells of a region for `OCloudNode`), then marks a
cell-ROP **positive** if an alarm is active within the next `horizon` ROPs. So
the task is genuine *prediction*: the model must read the precursor, not the
alarm itself. The 24 raw `probableCause` values are grouped into 8 fault classes
(+ Normal).

## Multi-level fusion

The cell is the primary entity. Each cell window carries its own RAN counters
(`du_*`, `cu_*`) plus its gNB's container metrics (`plat_*`) and its region's
host metrics (`infra_*`), joined via the `fdn` topology. Hosts are a regional
O-Cloud pool (not per-gNB), so infra is attached per region - a documented
approximation.

## Running

```bash
# smoke run on a laptop (CPU/MPS), a few days and gNBs
python -m autofm.pipeline --compressor pca --max-days 6 --max-gnbs 12

# the autofm design (MAE compressor)
python -m autofm.pipeline --compressor mae --max-days 14 --max-gnbs 40

# full config-driven run (edit config_autofm.yaml, widen knobs on Leonardo)
python -m autofm.pipeline --config config_autofm.yaml
```

### Config knobs (`config_autofm.yaml`)

| Key | Meaning |
|-----|---------|
| `max_days` / `max_gnbs` | data scale; raise toward 182 days / ~100 gNBs for full runs |
| `horizon` | forecast lead in ROPs (5 = 75 min ahead) |
| `window_k` | lookback window length in ROPs |
| `latent_dim` | compressed per-ROP latent size |
| `compressor` | `pca` (fast baseline) or `mae` (autofm design) |
| `mae_epochs` / `clf_epochs` | training lengths |
| `seed` | reproducibility |

## Status and next steps

This is the **baseline chain**, runnable end to end. It intentionally starts
simple so each richer component can be added and measured against it:

- [x] Multi-level loader + alarm labeling + leakage-safe windowing
- [x] Compression (PCA baseline + MAE) and a temporal Transformer classifier
- [ ] GAT topology fusion over the gNB/region graph (localization + causality)
- [ ] Evidential heads (uncertainty) + conformal calibration (coverage)
- [ ] Horizon sweep (m = 5..120) and the `falcon` baseline comparison

Current smoke numbers are a weak-but-honest baseline (see the per-class report
in a run): `Normal` is learned well, minority fault classes are not yet - the
expected headroom for MAE over PCA, more data, and the topology/uncertainty
components.
```
