"""Feature matrix, windowing, and leakage-safe splits.

Turns the labeled cell-centric frame (loader + labels) into model-ready tensors:

1. **Clean features** - drop all-null columns, forward/back-fill sparse pm
   counters per cell, fill any remainder with 0.
2. **Chronological split** - 60/20/20 by ROP time; val and test strictly after
   train (no shuffling across the boundary).
3. **Standardize** - per-feature mean/std fit on the **train split only**, then
   applied to val/test (the design-doc leakage rule).
4. **Window** - per cell, sliding ``k``-ROP lookback windows -> ``(N, k, F)``.
   The label is the window's final-step ``will_fault`` / ``fault_class`` (which
   already encodes "within the next ``horizon`` ROPs", set at label time).

The windower never crosses a cell boundary, so a window is always one cell's
contiguous history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TS = "rdiTimestampTZ"


@dataclass
class Dataset:
    """Windowed tensors for one split."""

    X: np.ndarray            # (N, k, F) float32
    y_binary: np.ndarray     # (N,) int8   - will_fault
    y_class: np.ndarray      # (N,) int64  - fault_class
    cells: np.ndarray        # (N,) object - originating cell id (for grouping)


@dataclass
class Prepared:
    """All splits + the fitted scaling stats and feature names."""

    train: Dataset
    val: Dataset
    test: Dataset
    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray


def _clean_features(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop all-null columns; fill sparse counters per cell, then zero-fill."""
    f = frame.copy()
    # Drop columns that are entirely null across the loaded data.
    keep = [c for c in feature_cols if f[c].notna().any()]
    # Per-cell forward then backward fill (dense 15-min telemetry), then 0.
    # (vectorised groupby fills; same result as a per-column ffill().bfill())
    f[keep] = f.groupby("cell")[keep].ffill()
    f[keep] = f.groupby("cell")[keep].bfill()
    f[keep] = f[keep].fillna(0.0)
    return f, keep


def _time_split_bounds(timestamps: np.ndarray,
                       train_frac: float, val_frac: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (train_end, val_end) timestamps for a chronological split.

    Indices are clamped to ``[0, n-1]`` so that very small numbers of distinct
    ROPs (e.g. smoke-scale runs) never underflow to ``-1`` and silently wrap to
    the last timestamp, which would corrupt the split boundaries.
    """
    ts_sorted = np.sort(np.unique(timestamps))
    n = len(ts_sorted)
    if n == 0:
        raise ValueError("No timestamps to split on")
    train_idx = min(max(int(n * train_frac) - 1, 0), n - 1)
    val_idx = min(max(int(n * (train_frac + val_frac)) - 1, 0), n - 1)
    train_end = ts_sorted[train_idx]
    val_end = ts_sorted[val_idx]
    return train_end, val_end


def _standardize_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[std == 0] = 1.0  # guard constant columns
    return mean.astype(np.float32), std.astype(np.float32)


@dataclass
class WindowRows:
    """One split as scaled per-ROP rows plus window offsets (no window copies).

    Window ``i`` is ``rows[starts[i] : starts[i] + k]``. Each ROP sits in up to
    ``k`` overlapping windows, so this is about ``k`` times smaller than the
    materialised ``(N, k, F)`` array it stands for.
    """

    rows: np.ndarray         # (R, F) float32 - scaled rows, one cell after another
    starts: np.ndarray       # (N,) int64     - first row of each window
    k: int
    y_binary: np.ndarray     # (N,) int8
    y_class: np.ndarray      # (N,) int64
    cells: np.ndarray        # (N,) object

    def row_index(self) -> np.ndarray:
        """Row of each step of each window, in window order: shape (N, k)."""
        return self.starts[:, None] + np.arange(self.k)

    def windows(self) -> np.ndarray:
        """The materialised ``(N, k, F)`` window array."""
        return self.rows[self.row_index()]


@dataclass
class PreparedRows:
    """Like :class:`Prepared`, with splits as :class:`WindowRows`."""

    train: WindowRows
    val: WindowRows
    test: WindowRows
    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray


def _window_rows(
    f: pd.DataFrame, feature_cols: list[str], k: int,
    mean: np.ndarray, std: np.ndarray,
) -> WindowRows:
    """Scale each cell's time-sorted rows and record where its windows start."""
    vals_l, starts_l, yb, yc, cs = [], [], [], [], []
    offset = 0
    for cell, g in f.groupby("cell", sort=False):
        g = g.sort_values(TS)
        n = len(g)
        if n < k:
            continue
        vals_l.append((g[feature_cols].to_numpy(dtype=np.float32) - mean) / std)
        # window = rows [end-k, end) for end in k..n; label at its last step
        starts_l.append(offset + np.arange(n - k + 1, dtype=np.int64))
        yb.append(g["will_fault"].to_numpy(dtype=np.int8)[k - 1:])
        yc.append(g["fault_class"].to_numpy(dtype=np.int64)[k - 1:])
        cs.append(np.full(n - k + 1, cell, dtype=object))
        offset += n
    if not vals_l:
        return WindowRows(np.empty((0, len(feature_cols)), np.float32),
                          np.empty(0, np.int64), k, np.empty(0, np.int8),
                          np.empty(0, np.int64), np.empty(0, object))
    return WindowRows(
        rows=np.concatenate(vals_l).astype(np.float32, copy=False),
        starts=np.concatenate(starts_l), k=k,
        y_binary=np.concatenate(yb), y_class=np.concatenate(yc),
        cells=np.concatenate(cs),
    )


def _make_windows(
    f: pd.DataFrame, feature_cols: list[str], k: int,
    mean: np.ndarray, std: np.ndarray,
) -> Dataset:
    """Build (N, k, F) windows per cell from a time-sorted frame slice."""
    w = _window_rows(f, feature_cols, k, mean, std)
    return Dataset(X=w.windows(), y_binary=w.y_binary, y_class=w.y_class,
                   cells=w.cells)


def _split_and_scale(labeled_frame, feature_cols, train_frac, val_frac):
    """Clean, split by time and fit the scaler on train rows only."""
    f, keep = _clean_features(labeled_frame, feature_cols)
    f = f.sort_values(["cell", TS]).reset_index(drop=True)

    train_end, val_end = _time_split_bounds(
        f[TS].to_numpy(), train_frac, val_frac)

    train_f = f[f[TS] <= train_end]
    val_f = f[(f[TS] > train_end) & (f[TS] <= val_end)]
    test_f = f[f[TS] > val_end]

    # Fit scaler on train rows only (leakage-safe).
    mean, std = _standardize_fit(train_f[keep].to_numpy(dtype=np.float32))
    return (train_f, val_f, test_f), keep, mean, std


def prepare(
    labeled_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int = 60,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Prepared:
    """Clean, split (by time), scale (train-fit), and window the labeled frame.

    Args:
        labeled_frame: output of ``label_frame`` (has will_fault, fault_class).
        feature_cols:  numeric feature columns to use as model inputs.
        k:             lookback window length in ROPs.
        train_frac/val_frac: chronological split fractions (test is the rest).
    """
    splits, keep, mean, std = _split_and_scale(
        labeled_frame, feature_cols, train_frac, val_frac)
    train, val, test = (_make_windows(s, keep, k, mean, std) for s in splits)
    return Prepared(train=train, val=val, test=test, feature_cols=keep,
                    mean=mean, std=std)


def prepare_rows(
    labeled_frame: pd.DataFrame,
    feature_cols: list[str],
    *,
    k: int = 60,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> PreparedRows:
    """:func:`prepare`, but splits stay as rows + window offsets.

    ``prepare_rows(...).train.windows()`` equals ``prepare(...).train.X``; use
    this when a consumer can index rows itself (e.g. the per-ROP MAE) to avoid
    holding ``k`` copies of every row.
    """
    splits, keep, mean, std = _split_and_scale(
        labeled_frame, feature_cols, train_frac, val_frac)
    train, val, test = (_window_rows(s, keep, k, mean, std) for s in splits)
    return PreparedRows(train=train, val=val, test=test, feature_cols=keep,
                        mean=mean, std=std)
