#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed Month-to-Symbol Rotation — NO LOOKAHEAD (IDLE-ready)

Rule: For each strategy, assign a single symbol to each calendar month (1..12).
Execution: On the strategy's intersection calendar, on each day use the daily
close→close return of the symbol assigned to that day’s month. Fully invested (100%).
Benchmark: continuous SPY buy & hold on the same calendar for plotting.

Outputs (./fixed_month_rotation_outputs):
  • <name>_detail.csv  (Y, M, Held, AppliedRet, Equity)
  • <name>_equity.csv
  • <name>_equity_vs_SPY.png
  • summary_metrics.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# =============== User Settings ===============
DATA_FOLDER = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR = "./fixed_month_rotation_outputs"
PLOT_START = "2010-01-01"      # rebasing date for plots
START_EQUITY = 100_000.0
BENCHMARK = "SPY"
VERBOSE = True

# ------------- Strategy Definitions -------------
# Each strategy is a dict:
#   name: str
#   buckets: list of (symbol, months_list) where months_list are ints in 1..12
#   All 12 months should be covered exactly once across the buckets.
STRATEGIES = [
    # 1) GLD,SMH. Hold GLD in [1,4,8,10,12] and SMH in the rest
    {
        "name": "GLD_SMH_fixed",
        "buckets": [
            ("GLD", [1,4,8,10,12]),
            ("SMH", [2,3,5,6,7,9,11]),
        ],
    },
    # 2) GLD,XLU,SMH - GLD[1,8,9], XLU[3,4,10], SMH[2,5,6,7,11,12]
    {
        "name": "GLD_XLU_SMH_fixed",
        "buckets": [
            ("GLD", [1,8,9]),
            ("XLU", [3,4,10]),
            ("SMH", [2,5,6,7,11,12]),
        ],
    },
    # 3) QQQ,GLD,SMH - [3,7,10], [1,4,8,9] , [2,5,6,11,12]
    {
        "name": "QQQ_GLD_SMH_fixed",
        "buckets": [
            ("QQQ", [3,7,10]),
            ("GLD", [1,4,8,9]),
            ("SMH", [2,5,6,11,12]),
        ],
    },
    # 4) SPY,QQQ,GLD - [4,11], [3,5,6,7,10], [1,2,8,9,12]
    {
        "name": "SPY_QQQ_GLD_fixed",
        "buckets": [
            ("SPY", [4,11]),
            ("QQQ", [3,5,6,7,10]),
            ("GLD", [1,2,8,9,12]),
        ],
    },
    # 5) QQQ,GLD,IWM - [3,4,5,7], [1,2,8,9], [6,10,11,12]
    {
        "name": "QQQ_GLD_IWM_fixed",
        "buckets": [
            ("QQQ", [3,4,5,7]),
            ("GLD", [1,2,8,9]),
            ("IWM", [6,10,11,12]),
        ],
    },
    # 6) GLD,SLV,SMH,USO - [4,8],[1,10],[2,3,5,7,9,11,12], [6]
    {
        "name": "GLD_SLV_SMH_USO_fixed",
        "buckets": [
            ("GLD", [4,8]),
            ("SLV", [1,10]),
            ("SMH", [2,3,5,7,9,11,12]),
            ("USO", [6]),
        ],
    },
    # 7) GLD,SMH,USO - [1,4,8], [2,3,5,7,9,10,11], [6,12]
    {
        "name": "GLD_SMH_USO_fixed",
        "buckets": [
            ("GLD", [1,4,8]),
            ("SMH", [2,3,5,7,9,10,11]),
            ("USO", [6,12]),
        ],
    },
    # 8) SMH,SLV - [2..12], [1]
    {
        "name": "SMH_SLV_fixed",
        "buckets": [
            ("SMH", [2,3,4,5,6,7,8,9,10,11,12]),
            ("SLV", [1]),
        ],
    },
    # 9) SLV,QQQ - [12,1,2], [3..11]
    {
        "name": "SLV_QQQ_fixed",
        "buckets": [
            ("SLV", [12,1,2]),
            ("QQQ", [3,4,5,6,7,8,9,10,11]),
        ],
    },
    # 10) TLT,SMH - [8,9], [rest]
    {
        "name": "TLT_SMH_fixed",
        "buckets": [
            ("TLT", [8,9]),
            ("SMH", [1,2,3,4,5,6,7,10,11,12]),
        ],
    },
    # 11) QQQ,SMH,SLV - [3..10], [11,12], [1,2]
    {
        "name": "QQQ_SMH_SLV_fixed",
        "buckets": [
            ("QQQ", [3,4,5,6,7,8,9,10]),
            ("SMH", [11,12]),
            ("SLV", [1,2]),
        ],
    },
    # 12) JNK,SMH,XHB - [9],[10,11,12,1,2,3,4,5,6],[7,8]
    {
        "name": "JNK_SMH_XHB_fixed",
        "buckets": [
            ("JNK", [9]),
            ("SMH", [10,11,12,1,2,3,4,5,6]),
            ("XHB", [7,8]),
        ],
    },
    # 13) QQQ,SMH,USO - [7,8,9,10], [11,12,1,2,3,4,5], [6]
    {
        "name": "QQQ_SMH_USO_fixed",
        "buckets": [
            ("QQQ", [7,8,9,10]),
            ("SMH", [11,12,1,2,3,4,5]),
            ("USO", [6]),
        ],
    },
]

# =============== IO Helpers ===============
def _coerce_datetime(df: pd.DataFrame) -> pd.DatetimeIndex:
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        first = df.columns[0]
        dt = pd.to_datetime(df[first], errors="coerce")
        if dt.isna().all():
            raise ValueError("No 'Date' column and first column not parseable")
        return pd.DatetimeIndex(dt)
    date_col = cols["date"]
    if "time" in cols:
        time_col = cols["time"]
        dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    if dt.isna().all():
        raise ValueError("Date/Time columns could not be parsed")
    return pd.DatetimeIndex(dt)

def read_close_daily(path: Path) -> pd.Series:
    df = None
    for sep in [None, ",", "\t", r"\s+"]:
        try:
            t = pd.read_csv(path, sep=sep, engine="python")
            if len(t):
                df = t; break
        except Exception:
            pass
    if df is None or df.empty:
        raise ValueError(f"Could not read: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    idx = _coerce_datetime(df)
    keep = ~idx.isna()
    df = df.loc[keep].copy(); idx = idx[keep]
    df.index = idx

    lower = {c.lower(): c for c in df.columns}
    close_col = None
    for cand in ("close","Close","CLOSE","settle","Settle","last","Last",
                 "adj_close","Adj Close","AdjClose","adjusted close"):
        if cand in df.columns or cand.lower() in lower:
            close_col = cand if cand in df.columns else lower[cand.lower()]
            break
    if close_col is None:
        raise ValueError(f"No close-like column in {path}")

    s = pd.to_numeric(df[close_col], errors="coerce").dropna()
    # collapse intraday to daily: last close per calendar day
    s = s.groupby(s.index.normalize()).last().sort_index().astype("float64")
    return s

def load_series(folder: str, sym: str) -> Optional[pd.Series]:
    p = Path(folder) / f"{sym}Raw.txt"
    try:
        return read_close_daily(p)
    except Exception as e:
        if VERBOSE:
            print(f"[WARN] {sym}: {e}")
        return None

# =============== Core Simulator ===============
def simulate_fixed_month_rotation(
    closes: Dict[str, pd.Series],
    mapping_month_to_symbol: Dict[int, str],
    benchmark: Optional[pd.Series],
    start_equity: float = START_EQUITY,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    For each date, pick the symbol assigned to that month and apply its daily return.
    Calendar is the intersection across all symbols in 'mapping_month_to_symbol'.
    """
    # Intersection calendar across all required symbols
    needed_syms = sorted(set(mapping_month_to_symbol.values()))
    idx = None
    for s in needed_syms:
        ser = closes[s]
        idx = ser.index if idx is None else idx.intersection(ser.index)
    if idx is None or len(idx) < 200:
        raise RuntimeError("Not enough common dates across symbols.")

    idx = idx.sort_values()
    # precompute returns for each symbol on the common calendar
    ret_map: Dict[str, pd.Series] = {}
    for s in needed_syms:
        r = closes[s].reindex(idx).pct_change().fillna(0.0)
        # clip extreme negatives to avoid numeric blow-ups
        ret_map[s] = np.maximum(r.to_numpy(), -0.999999)

    # For each day, pick the assigned symbol by month and apply its daily return
    months = idx.month
    applied = np.zeros(len(idx), dtype=float)
    held = np.empty(len(idx), dtype=object)

    # vectorized pass per month
    for m in range(1, 13):
        if m not in mapping_month_to_symbol:
            continue
        sym = mapping_month_to_symbol[m]
        mask = (months == m)
        if mask.any():
            applied[mask] = ret_map[sym][mask]
            held[mask] = sym

    # Equity
    eq_vals = (1.0 + applied).cumprod() * float(start_equity)
    eq = pd.Series(eq_vals, index=idx, name="Equity")

    # Detail dataframe
    detail = pd.DataFrame({
        "Y": idx.year,
        "M": months,
        "Held": held,
        "AppliedRet": applied,
        "Equity": eq_vals,
    }, index=idx)

    # Benchmark equity on same calendar
    bench_eq = pd.Series(dtype="float64")
    if benchmark is not None:
        b = benchmark.reindex(idx).ffill()
        br = b.pct_change().fillna(0.0).to_numpy()
        br = np.maximum(br, -0.999999)
        bench_eq = pd.Series((1.0 + br).cumprod() * float(start_equity), index=idx, name="Benchmark_Equity")

    return eq, detail, bench_eq

# =============== Metrics & Plotting ===============
def metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 50:
        return {"CAGR": np.nan, "TotalRet": np.nan, "MaxDD": np.nan}
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days, 1) / 365.25
    cagr = float(np.exp(np.log(eq.iloc[-1] / eq.iloc[0]) / years) - 1.0) if eq.iloc[0] > 0 and years > 0 else np.nan
    peak = np.maximum.accumulate(eq.to_numpy())
    dd = eq.to_numpy() / peak - 1.0
    return {"CAGR": cagr, "TotalRet": total, "MaxDD": float(dd.min())}

def plot_equity(name: str, eq: pd.Series, bench: pd.Series, outdir: Path, plot_start: str):
    import matplotlib.pyplot as plt
    ps = pd.Timestamp(plot_start)
    eqp = eq[eq.index >= ps]
    if len(eqp) < 2:
        return
    eqp = eqp / float(eqp.iloc[0]) * START_EQUITY

    bqp = pd.Series(dtype=float)
    if isinstance(bench, pd.Series) and not bench.empty:
        bqp = bench[bench.index >= ps]
        if len(bqp) >= 1:
            bqp = bqp / float(bqp.iloc[0]) * START_EQUITY
            cal = eqp.index.union(bqp.index)
            eqp = eqp.reindex(cal, method="ffill")
            bqp = bqp.reindex(cal, method="ffill")

    fig = plt.figure(figsize=(12,5)); ax = fig.add_subplot(111)
    if not bqp.empty and bqp.notna().sum() >= 2:
        ax.plot(bqp.index, bqp.values, label=f"{BENCHMARK} B&H", linestyle="--", linewidth=1.6, zorder=1)
    ax.plot(eqp.index, eqp.values, label=name, linewidth=1.8, zorder=2)
    ax.set_title(f"{name} (rebased @ {ps.date()})", fontsize=10)
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(outdir / f"{name}_equity_vs_{BENCHMARK}.png", dpi=130)
    plt.close(fig)

# =============== Runner ===============
def build_month_map(buckets: list[tuple[str, list[int]]]) -> dict[int, str]:
    """Validate coverage and build month -> symbol map."""
    month_map: dict[int, str] = {}
    for sym, months in buckets:
        for m in months:
            if m < 1 or m > 12:
                raise ValueError(f"Invalid month {m} for {sym}")
            if m in month_map:
                raise ValueError(f"Month {m} assigned twice ({month_map[m]} and {sym})")
            month_map[m] = sym
    missing = [m for m in range(1,13) if m not in month_map]
    if missing:
        raise ValueError(f"Months not assigned: {missing}")
    return month_map

def run_all():
    outp = Path(OUTDIR); outp.mkdir(parents=True, exist_ok=True)

    # Determine all symbols we need
    needed = {BENCHMARK}
    for s in STRATEGIES:
        for sym, _months in s["buckets"]:
            needed.add(sym)

    # Load closes
    closes: Dict[str, pd.Series] = {}
    for sym in sorted(needed):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable data for {sym}")
        closes[sym] = ser.sort_index()

    # Loop strategies
    rows = []
    for strat in STRATEGIES:
        name = strat["name"]
        month_map = build_month_map(strat["buckets"])
        try:
            eq, detail, bench_eq = simulate_fixed_month_rotation(
                closes, month_map, closes.get(BENCHMARK), start_equity=START_EQUITY
            )
            # Save CSVs
            detail.to_csv(Path(OUTDIR) / f"{name}_detail.csv", index=True)
            eq.rename("Equity").to_csv(Path(OUTDIR) / f"{name}_equity.csv", header=True)
            # Plot
            plot_equity(name, eq, bench_eq, Path(OUTDIR), PLOT_START)
            # Metrics
            ms = metrics_from_equity(eq)
            rows.append({"Name": name, "CAGR": ms["CAGR"], "TotalRet": ms["TotalRet"], "MaxDD": ms["MaxDD"]})
            if VERBOSE:
                print(f"[OK] {name}: CAGR={ms['CAGR']:.3%}  Total={ms['TotalRet']:.2%}  MaxDD={ms['MaxDD']:.2%}")
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(Path(OUTDIR) / "summary_metrics.csv", index=False)
        print(f"\nDone. Outputs → {Path(OUTDIR).resolve()}")
    else:
        print("No strategies produced output. Check data coverage / symbol names.")

# =============== Main ===============
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run_all()
