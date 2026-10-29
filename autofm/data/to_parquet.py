"""Convert the CSV dataset to Parquet once, so loading is fast.

Reading the ``;``-delimited CSVs is the slow part of every run: pandas parses
text and the loader then coerces every counter column to a number. This writes
the same files as typed Parquet (meta columns as strings, counters as float64)
under a mirror of the dataset layout::

    <src>/2025/<MM>/<level>/<date>/<group>_<level>_<date>.csv
 -> <dst>/2025/<MM>/<level>/<date>/<group>_<level>_<date>.parquet
    <src>/alarms/alarms.csv -> <dst>/alarms/alarms.csv  (copied as-is)

Point ``dataset_root`` at ``<dst>`` and :func:`autofm.data.loader.load` picks
the Parquet files up; the CSV dataset is only read, never modified.

Usage::

    python -m autofm.data.to_parquet --src <csv dataset> --dst <parquet dataset>
    python -m autofm.data.to_parquet ... --max-days 14 --workers 8
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from autofm.profile_dataset import META_COLS

# (level, group) pairs the loader reads; pass --groups to convert others too.
DEFAULT_GROUPS = [
    ("pm", "NRCellDU"),
    ("pm", "NRCellCU"),
    ("platform", "ManagedNFService"),
    ("infra", "OCloudNode"),
]


def csv_to_frame(path: Path) -> pd.DataFrame:
    """Read one dataset CSV with the same typing the loader applies."""
    df = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    for c in df.columns:
        if c not in META_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _convert_one(src: Path, dst: Path) -> tuple[str, int, float]:
    """Convert one CSV; write to a temp name first so a crash leaves no half file."""
    t0 = time.perf_counter()
    df = csv_to_frame(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, dst)
    return str(dst), len(df), time.perf_counter() - t0


def plan(src: Path, dst: Path, groups: list[tuple[str, str]],
         max_days: int | None) -> list[tuple[Path, Path]]:
    """List (csv, parquet) pairs still to convert, oldest days first."""
    dates = sorted({p.name for p in src.glob("2025/*/pm/*") if p.is_dir()})
    if max_days is not None:
        dates = dates[:max_days]
    keep = set(dates)
    jobs = []
    for level, group in groups:
        for day in sorted(src.glob(f"2025/*/{level}/*")):
            if day.name not in keep:
                continue
            # skip macOS "._" resource-fork files that ship next to the CSVs
            for csv in sorted(day.glob(f"{group}_{level}_*.csv")):
                out = dst / csv.relative_to(src).with_suffix(".parquet")
                if not out.exists():
                    jobs.append((csv, out))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="CSV dataset root")
    ap.add_argument("--dst", required=True, help="Parquet dataset root (created)")
    ap.add_argument("--max-days", type=int, default=None,
                    help="convert only the first N days (default: all)")
    ap.add_argument("--groups", nargs="*", default=None, metavar="LEVEL:GROUP",
                    help="e.g. pm:S1ULink; default: the groups the loader reads")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    groups = ([tuple(g.split(":", 1)) for g in args.groups]
              if args.groups else DEFAULT_GROUPS)

    alarms = src / "alarms" / "alarms.csv"
    if alarms.exists():
        (dst / "alarms").mkdir(parents=True, exist_ok=True)
        shutil.copy2(alarms, dst / "alarms" / "alarms.csv")

    jobs = plan(src, dst, groups, args.max_days)
    print(f"[parquet] {len(jobs)} files to convert with {args.workers} workers "
          f"(already converted files are skipped)", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_convert_one, s, d) for s, d in jobs]
        for i, f in enumerate(futures, 1):
            out, rows, sec = f.result()
            print(f"[parquet] {i}/{len(jobs)} {Path(out).name} "
                  f"{rows} rows {sec:.1f}s", flush=True)
    print(f"[parquet] done in {time.perf_counter() - t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    main()
