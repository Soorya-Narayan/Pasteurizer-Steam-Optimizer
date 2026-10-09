"""Load, de-duplicate and clean the raw pasteurizer SCADA CSV exports."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def sanitize_column(name: str) -> str:
    """'TT-06 Holding Out1 (°C)' -> 'tt_06_holding_out1_c' (safe for LightGBM/XGBoost)."""
    s = name.lower().replace("°", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_unique_csvs(csv_paths: list[Path]) -> tuple[list[Path], list[list[Path]]]:
    """Group byte-identical files; return (unique files, duplicate groups)."""
    groups: dict[str, list[Path]] = {}
    for p in sorted(csv_paths, key=_natural_key):
        groups.setdefault(_md5(p), []).append(p)
    unique = [g[0] for g in groups.values()]
    dupes = [g for g in groups.values() if len(g) > 1]
    return unique, dupes


def clean_frame(df: pd.DataFrame, timestamp_col: str = "date_time") -> pd.DataFrame:
    """Sanitize names, parse types, strip quotes, sort & de-duplicate timestamps."""
    df = df.copy()
    df.columns = [sanitize_column(c) for c in df.columns]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])

    for c in df.columns:
        if c == timestamp_col or pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = df[c].astype("string").str.strip().str.strip('"').replace({"--": pd.NA, "": pd.NA})
        numeric = pd.to_numeric(s, errors="coerce")
        n_present = int(s.notna().sum())
        if n_present and numeric.notna().sum() >= 0.9 * n_present:
            df[c] = numeric
        else:
            df[c] = s

    df = df.dropna(axis=1, how="all")  # disconnected channels (TT-13 / TT-14 ...)
    df = df.sort_values(timestamp_col).drop_duplicates(subset=timestamp_col)
    return df.reset_index(drop=True)


def read_csvs(paths: list[Path], timestamp_col: str = "date_time", progress: bool = True) -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in tqdm(paths, desc="Reading CSVs", unit="file", disable=not progress)]
    return clean_frame(pd.concat(frames, ignore_index=True), timestamp_col)


def load_dataset(cfg: dict, rebuild: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Return the cleaned, chronologically sorted dataset (cached as parquet)."""
    dcfg = cfg["data"]
    cache = Path(dcfg["cache_path"])
    ts = dcfg["timestamp_col"]

    if cache.exists() and not rebuild:
        df = pd.read_parquet(cache)
        if ts in df.columns:
            if verbose:
                print(f"  • Loaded cached dataset {cache} -> {len(df):,} rows")
            return df
        if verbose:
            print("  • Cache schema is outdated – rebuilding")

    paths = list(Path(dcfg["csv_dir"]).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {dcfg['csv_dir']}")
    unique, dupes = discover_unique_csvs(paths)
    if verbose:
        print(f"  • Found {len(paths)} CSVs -> {len(unique)} unique ({sum(len(g) - 1 for g in dupes)} byte-identical duplicates skipped)")

    df = read_csvs(unique, ts, progress=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    if verbose:
        print(f"  • Cleaned dataset: {len(df):,} rows x {df.shape[1]} cols  ({df[ts].min()} -> {df[ts].max()})")
        print(f"  • Cached to {cache}")
    return df
