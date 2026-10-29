"""Alarm-to-label layer.

Telemetry carries no inline fault label. Labels come from ``alarms.csv``: each
alarm is an interval ``[eventTime, clearTime]`` on an ``fdn``. We turn that feed
into a supervised **fault-prediction** target on the cell-centric telemetry
frame produced by :mod:`autofm.data.loader`.

Two steps:

1. **Resolve scope.** Map each alarm's ``fdn`` to the cells it affects, using
   the same topology the loader uses:

   * ``NRCellDU``        -> that exact cell (``gnb`` + ``cell`` id).
   * ``S1ULink``         -> all cells of the alarm's gNB (``MeContext``).
   * ``ManagedNFService``-> all cells of the alarm's gNB (service name prefix).
   * ``OCloudNode``      -> all cells in the alarm's region (SubNetwork).

2. **Forecast label.** A cell at ROP ``t`` is **positive** if any in-scope
   alarm **onsets** (``eventTime``) within the next ``horizon`` ROPs, i.e. its
   start falls in ``(t, t + horizon]`` (``t`` exclusive). An alarm already
   firing at or before ``t`` does **not** make the row positive - that would be
   detecting a present fault, not forecasting a future one (label leakage).
   This keeps the task genuine forecasting: the model must read the precursor,
   not the alarm itself. We emit both a binary ``will_fault`` label and a
   multi-class ``fault_class`` (grouped probable cause; ``0`` = Normal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autofm.data.loader import _region_of
from autofm.fdn import parse_fdn

ROP = pd.Timedelta(minutes=15)

# Group the 24 raw probableCause values into a compact class taxonomy. This
# keeps the first classifier tractable and maps causes to the level they most
# naturally belong to. 0 is reserved for Normal (no imminent alarm).
CAUSE_TO_CLASS: dict[str, str] = {
    # compute / host pressure
    "congestion": "Resource",
    "outOfMemory": "Resource",
    "storageCapacityProblem": "Resource",
    "highTemperature": "Hardware",
    "coolingSystemFailure": "Hardware",
    "powerProblem": "Hardware",
    "equipmentMalfunction": "Hardware",
    "replaceableUnitProblem": "Hardware",
    # radio / signal
    "degradedSignal": "Radio",
    "lossOfSignal": "Radio",
    "lossOfFrame": "Radio",
    "lossOfSynchronisation": "Radio",
    "performanceDegraded": "Performance",
    "thresholdCrossed": "Performance",
    # transport / connectivity
    "lanError": "Transport",
    "connectionEstablishmentError": "Transport",
    # software / config / security
    "softwareError": "Software",
    "configurationError": "Software",
    "proceduralError": "Software",
    "informationModificationDetected": "Security",
    "authenticationFailure": "Security",
    "keyExpired": "Security",
    "cableTamper": "Security",
    "indeterminate": "Other",
}

CLASS_NAMES = ["Normal", "Resource", "Hardware", "Radio", "Performance",
               "Transport", "Software", "Security", "Other"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class ResolvedAlarm:
    """An alarm mapped to a cell scope + a time interval + a class id."""

    scope_type: str           # "cell" | "gnb" | "region"
    gnb: str | None
    cell: str | None
    region: str | None
    start: pd.Timestamp
    end: pd.Timestamp
    class_id: int


def _resolve_scope(fdn: str) -> tuple[str, str | None, str | None, str | None]:
    """Return (scope_type, gnb, cell, region) for an alarm fdn."""
    comps = parse_fdn(fdn)
    obj = list(comps.keys())[-1] if comps else None
    region = _region_of(fdn)
    if obj == "NRCellDU":
        return "cell", comps.get("MeContext"), comps.get("NRCellDU"), region
    if obj == "S1ULink":
        return "gnb", comps.get("MeContext"), None, region
    if obj == "ManagedNFService":
        svc = comps.get("ManagedNFService", "")
        return "gnb", svc.split("-", 1)[0] or None, None, region
    if obj == "OCloudNode":
        return "region", None, None, region
    return "region", None, None, region  # default: widest scope


def load_alarms(alarms_csv: str) -> list[ResolvedAlarm]:
    """Read and resolve every alarm into a cell-scoped interval + class."""
    a = pd.read_csv(alarms_csv, sep=";", dtype=str)
    a["eventTime"] = pd.to_datetime(a["eventTime"], utc=True, errors="coerce")
    a["clearTime"] = pd.to_datetime(a["clearTime"], utc=True, errors="coerce")
    # An alarm with no clear time is treated as open until end of its day + 1d.
    a["clearTime"] = a["clearTime"].fillna(a["eventTime"] + pd.Timedelta(days=1))

    resolved: list[ResolvedAlarm] = []
    for _, r in a.iterrows():
        if pd.isna(r["eventTime"]):
            continue
        st, gnb, cell, region = _resolve_scope(r["fdn"])
        cls = CAUSE_TO_CLASS.get(r.get("probableCause"), "Other")
        resolved.append(ResolvedAlarm(
            scope_type=st, gnb=gnb, cell=cell, region=region,
            start=r["eventTime"], end=r["clearTime"],
            class_id=CLASS_TO_ID[cls],
        ))
    return resolved


def label_frame(
    frame: pd.DataFrame,
    alarms: list[ResolvedAlarm],
    *,
    horizon: int = 5,
    ts_col: str = "rdiTimestampTZ",
    mode: str = "forecast",
) -> pd.DataFrame:
    """Add ``will_fault`` (0/1) and ``fault_class`` (int) to the telemetry frame.

    A row (gnb, cell, t) is positive if an in-scope alarm **onsets** in
    ``(t, t + horizon*ROP]`` - its ``eventTime`` is strictly after ``t`` and no
    later than ``t + horizon``. Alarms already active at or before ``t`` are
    ignored (predicting the future, not detecting the present). When several
    alarms qualify, the earliest-onsetting one's class wins (the imminent
    fault). Normal rows get class 0.

    Args:
        frame:   cell-centric telemetry frame (must have gnb, cell, region, ts).
        alarms:  resolved alarms from :func:`load_alarms`.
        horizon: forecast horizon in ROPs (15-min steps).
        mode:    ``"forecast"`` (default, the rule above) or ``"active"``: a
                 row is positive while an in-scope alarm is active,
                 ``start <= t < end`` (detection, not forecasting). ``horizon``
                 is then unused.
    """
    if mode not in ("forecast", "active"):
        raise ValueError(f"label mode must be 'forecast' or 'active', got {mode!r}")
    frame = frame.copy()
    # Both telemetry and alarms are UTC. Drop tz to compare as plain datetime64
    # (numpy comparisons below are tz-naive); this is lossless since all UTC.
    ts_naive = frame[ts_col].dt.tz_convert("UTC").dt.tz_localize(None)
    t = ts_naive.to_numpy()
    will = np.zeros(len(frame), dtype=np.int8)
    cls = np.zeros(len(frame), dtype=np.int64)
    # Earliest alarm start seen per row, so the nearest fault's class wins.
    best_start = np.full(len(frame), np.datetime64("2100-01-01T00:00:00"), dtype="datetime64[ns]")

    horizon_td = horizon * ROP

    # Pre-index rows by scope key for cheap matching: integer codes compare far
    # faster than the object (string) arrays, once per alarm over every row.
    # A key absent from the frame gets code -2, which matches no row (NaN -> -1).
    gnb_arr, gnb_keys = pd.factorize(frame["gnb"])
    cell_arr, cell_keys = pd.factorize(frame["cell"])
    region_arr, region_keys = pd.factorize(frame["region"])
    gnb_code = {k: i for i, k in enumerate(gnb_keys)}
    cell_code = {k: i for i, k in enumerate(cell_keys)}
    region_code = {k: i for i, k in enumerate(region_keys)}

    # Forecast window end per row (tz-naive, matching ``t``).
    win_end = (ts_naive + horizon_td).to_numpy()

    # Diagnostics: alarms whose scope matches no telemetry row, and alarms that
    # match rows but never overlap any forecast window. Both are silently
    # dropped from the labels, so surfacing their counts catches scope/topology
    # mismatches (e.g. an S1ULink gNB key that no cell shares).
    n_no_scope_match = 0
    n_no_window_overlap = 0

    for al in alarms:
        # Row matches the alarm's scope?
        if al.scope_type == "cell":
            mask = ((gnb_arr == gnb_code.get(al.gnb, -2))
                    & (cell_arr == cell_code.get(al.cell, -2)))
        elif al.scope_type == "gnb":
            mask = gnb_arr == gnb_code.get(al.gnb, -2)
        else:  # region
            mask = region_arr == region_code.get(al.region, -2)
        if not mask.any():
            n_no_scope_match += 1
            continue

        # Genuine FORECAST label: the alarm must *onset* strictly inside the
        # window (t, t + horizon]. An alarm that was already firing at or before
        # t is a concurrent/past event, not a future one, so it must NOT make
        # the row positive - otherwise the task degrades from "predict the
        # precursor" to "detect the alarm already present" (label leakage).
        #   positive  <=>  t < start <= t + horizon
        start = np.datetime64(al.start.to_datetime64())
        if mode == "active":
            end = np.datetime64(al.end.to_datetime64())
            overlap = mask & (start <= t) & (t < end)
        else:
            overlap = mask & (start > t) & (start <= win_end)
        if not overlap.any():
            n_no_window_overlap += 1
            continue

        will[overlap] = 1
        # Keep the class of the earliest-starting qualifying alarm.
        closer = overlap & (start < best_start)
        best_start[closer] = start
        cls[closer] = al.class_id

    total = len(alarms)
    n_applied = total - n_no_scope_match - n_no_window_overlap
    if n_no_scope_match or n_no_window_overlap:
        print(
            f"[labels] applied {n_applied}/{total} alarms "
            f"({n_no_scope_match} matched no in-scope rows, "
            f"{n_no_window_overlap} matched rows but no "
            f"{'active period' if mode == 'active' else 'forecast window'})"
        )

    frame["will_fault"] = will
    frame["fault_class"] = cls
    return frame


def label_from_config(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Load the alarm feed and label ``frame`` as the config asks.

    Config keys: ``horizon``; ``label_mode`` (``"forecast"``, default, or
    ``"active"``); ``alarm_scopes`` (optional subset of ``cell``, ``gnb``,
    ``region``: only alarms of those scopes make labels).
    """
    from pathlib import Path

    alarms = load_alarms(str(Path(cfg["dataset_root"]) / "alarms" / "alarms.csv"))
    scopes = cfg.get("alarm_scopes")
    if scopes:
        alarms = [a for a in alarms if a.scope_type in set(scopes)]
    return label_frame(frame, alarms, horizon=cfg["horizon"],
                       mode=cfg.get("label_mode", "forecast"))
