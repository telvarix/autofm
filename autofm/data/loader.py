"""Multi-level telemetry loader.

Loads the O-RAN dataset (infra / platform / pm) into a single tidy, time-indexed
frame keyed on ``(gnb, cell, timestamp)``. The **cell** is the primary entity:
it carries the richest counters (NRCellDU + NRCellCU) and the bulk of alarms.
Each cell is enriched with its gNB's **platform** (container) and **infra**
(host) metrics, following the ``fdn`` topology, so the model sees all three
levels at once - the whole point of the project.

Topology linking (derived from the fdn format in the dataset):

* Cell   ``...,MeContext=CIDU08,...,NRCellDU=CIDU1U``   -> gnb=``CIDU08`` cell=``CIDU1U``
* NF ctr ``...,ManagedNFService=CIDU08-GNBCUUPFunction`` -> gnb=``CIDU08`` (prefix)
* Host   ``...,ManagedElement=CORDU01,OCloudNode=CORDU01`` -> host=``CORDU01``

Cell and container share the **gNB** key. The host is linked via the cell's
``ManagedElement``. Platform rows for a gNB are aggregated (mean) across its NF
services; infra rows are taken per host.

Scale knobs keep local (CPU/MPS) runs cheap: load a handful of days and cap the
number of gNBs. Full-scale runs on Leonardo just widen those knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from autofm.fdn import META_COLS, parse_fdn, region_of

TS = "rdiTimestampTZ"  # the real 15-min ROP timestamp column


@dataclass
class LoadedData:
    """Tidy multi-level frame + metadata.

    * ``frame``       : one row per (gnb, cell, timestamp) with all level
                        features as columns (prefixed by level).
    * ``feature_cols``: names of the numeric feature columns (model inputs).
    * ``timestamps``  : sorted unique ROP timestamps present.
    """

    frame: pd.DataFrame
    feature_cols: list[str]
    timestamps: np.ndarray


def _gnb_of_cell(fdn: str) -> str | None:
    """gNB key for a cell fdn = its MeContext (falls back to ManagedElement)."""
    c = parse_fdn(fdn)
    return c.get("MeContext") or c.get("ManagedElement")


def _cell_of_cell(fdn: str) -> str | None:
    c = parse_fdn(fdn)
    return c.get("NRCellDU") or c.get("NRCellCU")


def _gnb_of_container(fdn: str) -> str | None:
    """gNB key for a platform NF fdn = prefix of ManagedNFService before '-'."""
    c = parse_fdn(fdn)
    svc = c.get("ManagedNFService")
    return svc.split("-", 1)[0] if svc else None


def _me_of_cell(fdn: str) -> str | None:
    return parse_fdn(fdn).get("ManagedElement")


def _region_of(fdn: str) -> str | None:
    """Region for an fdn = its third SubNetwork level (see :mod:`autofm.fdn`).

    Thin wrapper kept for the loader's internal call sites and for the labeler,
    which imports it. The region logic itself lives in :func:`autofm.fdn.region_of`
    so the loader, labeler, and profiler all agree.
    """
    return region_of(fdn)


def _day_dirs(root: Path, level: str) -> list[Path]:
    """All per-day directories for a level, sorted by date."""
    out: list[Path] = []
    for month in sorted(root.glob("2025/*")):
        d = month / level
        if d.is_dir():
            out.extend(sorted(p for p in d.glob("*") if p.is_dir()))
    return sorted(out, key=lambda p: p.name)


def _map_unique(s: pd.Series, fn) -> pd.Series:
    """``s.map(fn)``, but calling ``fn`` once per distinct value.

    Every fdn repeats once per ROP (96 times a day), so parsing only the
    distinct strings gives the same result for a fraction of the regex work.
    """
    return s.map({v: fn(v) for v in s.dropna().unique()})


def _gnbs_of_days(days: list[Path]) -> set[str]:
    """All gNBs that appear in the NRCellDU files of ``days`` (reads only fdn)."""
    gnbs: set[str] = set()
    for day in days:
        pq = list(day.glob("NRCellDU_pm_*.parquet"))
        if pq:
            fdn = pd.read_parquet(pq[0], columns=["fdn"])["fdn"]
        else:
            csv = list(day.glob("NRCellDU_pm_*.csv"))
            if not csv:
                continue
            fdn = pd.read_csv(csv[0], sep=";", dtype=str, usecols=["fdn"])["fdn"]
        gnbs.update(g for g in map(_gnb_of_cell, fdn.dropna().unique()) if g)
    return gnbs


def _read_group(day_dir: Path, group: str, level: str) -> pd.DataFrame | None:
    """Read one <group>_<level>_<date> file if present, numeric-coerced.

    Prefers the Parquet copy written by ``autofm.data.to_parquet`` (already
    typed, so no parsing or coercion) and falls back to the CSV.
    """
    matches = list(day_dir.glob(f"{group}_{level}_*.parquet"))
    if matches:
        return pd.read_parquet(matches[0])
    matches = list(day_dir.glob(f"{group}_{level}_*.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0], sep=";", dtype=str, low_memory=False)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def load(
    root: str | Path,
    *,
    max_days: int | None = 3,
    max_gnbs: int | None = 20,
    include_infra: bool = True,
    include_platform: bool = True,
) -> LoadedData:
    """Load a multi-level, cell-centric, time-indexed frame.

    Args:
        root:             dataset root (autofm-dataset-ie).
        max_days:         cap on number of days (None = all 182).
        max_gnbs:         cap on number of gNBs (None = all).
        include_infra:    attach host metrics.
        include_platform: attach container metrics.
    """
    root = Path(root).expanduser()
    cell_days = _day_dirs(root, "pm")
    if max_days is not None:
        cell_days = cell_days[:max_days]
    if not cell_days:
        raise FileNotFoundError(f"No pm/ day directories under {root}")

    # Pick the first N gNBs (stable, sorted) up front, so rows of other gNBs
    # are dropped per day instead of after loading everything.
    keep: set[str] | None = None
    if max_gnbs is not None:
        keep = set(sorted(_gnbs_of_days(cell_days))[:max_gnbs])

    cell_frames: list[pd.DataFrame] = []
    infra_frames: list[pd.DataFrame] = []
    plat_frames: list[pd.DataFrame] = []

    for day in cell_days:
        date = day.name
        # pm day dir is 2025/<MM>/pm/<date>; the sibling levels live under
        # 2025/<MM>/<level>/<date>. Resolve them relative to the pm dir.
        month_dir = day.parent.parent
        plat_day = month_dir / "platform" / date
        infra_day = month_dir / "infra" / date

        du = _read_group(day, "NRCellDU", "pm")
        cu = _read_group(day, "NRCellCU", "pm")
        if du is None:
            continue

        du = du.copy()
        du["gnb"] = _map_unique(du["fdn"], _gnb_of_cell)
        if keep is not None:
            du = du[du["gnb"].isin(keep)].copy()
        du["cell"] = _map_unique(du["fdn"], _cell_of_cell)
        du["me"] = _map_unique(du["fdn"], _me_of_cell)
        du["region"] = _map_unique(du["fdn"], _region_of)
        du_feats = _numeric_feature_cols(du)
        du = du.rename(columns={c: f"du_{c}" for c in du_feats})

        merged = du[["gnb", "cell", "me", "region", TS]
                     + [f"du_{c}" for c in du_feats]].copy()

        # Join the CU view of the same cell (same gnb + cell id).
        if cu is not None:
            cu = cu.copy()
            cu["gnb"] = _map_unique(cu["fdn"], _gnb_of_cell)
            if keep is not None:
                cu = cu[cu["gnb"].isin(keep)].copy()
            cu["cell"] = _map_unique(cu["fdn"], _cell_of_cell)
            cu_feats = _numeric_feature_cols(cu)
            cu = cu.rename(columns={c: f"cu_{c}" for c in cu_feats})
            merged = merged.merge(
                cu[["gnb", "cell", TS] + [f"cu_{c}" for c in cu_feats]],
                on=["gnb", "cell", TS], how="left",
            )
        cell_frames.append(merged)

        if include_platform:
            plat = _read_group(plat_day, "ManagedNFService", "platform")
            if plat is not None:
                plat = plat.copy()
                plat["gnb"] = _map_unique(plat["fdn"], _gnb_of_container)
                if keep is not None:
                    plat = plat[plat["gnb"].isin(keep)]
                pf = _numeric_feature_cols(plat)
                # Aggregate the gNB's NF services to one row per (gnb, ts).
                agg = (plat.groupby(["gnb", TS])[pf].mean()
                       .reset_index()
                       .rename(columns={c: f"plat_{c}" for c in pf}))
                plat_frames.append(agg)

        if include_infra:
            infra = _read_group(infra_day, "OCloudNode", "infra")
            if infra is not None:
                infra = infra.copy()
                infra["region"] = _map_unique(infra["fdn"], _region_of)
                inf = _numeric_feature_cols(infra)
                # Host names are a regional pool, not per-gNB, so aggregate the
                # region's hosts to one host-context vector per (region, ts).
                agg = (infra.groupby(["region", TS])[inf].mean()
                       .reset_index()
                       .rename(columns={c: f"infra_{c}" for c in inf}))
                infra_frames.append(agg)

    frame = pd.concat(cell_frames, ignore_index=True)


    # Attach platform (by gnb+ts) and infra (by region+ts - hosts are a
    # regional O-Cloud pool, not per-gNB, so the region is the reliable join).
    if plat_frames:
        plat_all = pd.concat(plat_frames, ignore_index=True)
        frame = frame.merge(plat_all, on=["gnb", TS], how="left")
    if infra_frames:
        infra_all = pd.concat(infra_frames, ignore_index=True)
        frame = frame.merge(infra_all, on=["region", TS], how="left")

    frame[TS] = pd.to_datetime(frame[TS], utc=True, errors="coerce")
    frame = frame.dropna(subset=[TS, "gnb", "cell"])
    frame = frame.sort_values(["gnb", "cell", TS]).reset_index(drop=True)

    feature_cols = [c for c in frame.columns
                    if c.startswith(("du_", "cu_", "plat_", "infra_"))]
    timestamps = np.sort(frame[TS].unique())

    return LoadedData(frame=frame, feature_cols=feature_cols, timestamps=timestamps)
