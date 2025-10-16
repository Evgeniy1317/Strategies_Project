#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week-of-Month Rotation (No Lookahead) — IDLE-ready

Rule:
  • For each calendar month, compute TDOM (trading day of month) = 1..N.
  • Trading week = 1 + floor((TDOM-1)/5). Clamp ≥4 → 4 (so week 4 and week 5 are the same).
  • Hold the symbol assigned to that trading week for that day. Fully invested (100%).
  • Execution uses daily close→close returns on the strategy's intersection calendar.
  • Benchmark: continuous SPY buy & hold on the same calendar.

Strategies:
  1) EEM, GLD, TLT, QQQ  → weeks 1,2,3,4/5
  2) XME, XLK, XLV, XLY  → weeks 1,2,3,4/5

Outputs (./week_rotation_outputs):
  • <name>_detail.csv  (Y, M, TDOM, Week, Held, AppliedRet, Equity)
  • <name>_equity.csv
  • <name>_equity_vs_SPY.png
  • summary_metrics.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# =========================
# User configuration
# =========================
DATA_FOLDER = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR = "./week_rotation_outputs"
PLOT_START = "2010-01-01"   # rebasing date for plots
START_EQUITY = 100_000.0
BENCHMARK = "SPY"
VERBOSE = True

# Strategies (week1, week2, week3, week4/5)
STRATEGIES: List[Tuple[str, List[str]]] = [
    ("EEM_GLD_TLT_QQQ_weeks", ["EEM", "GLD", "TLT", "QQQ"]),
    ("XME_XLK_XLV_XLY_weeks", ["XME", "XLK", "XLV", "XLY"]),
]

# =========================
# IO Helpers (robust loader)
# =========================
def _coerce_datetime(df: pd.DataFrame) -> pd.DatetimeIndex:
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        first = df.columns[0]
        dt = pd.to_datetime(df[first], errors="coerce")
        if dt.isna().all():
            raise ValueError("No 'Date' column and first column not parseable as dates")
        return pd.DatetimeIndex(dt)
    date_col = cols["date"]
    if "time" in cols:
        time_col = cols["time"]
        dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    else:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    if dt.isna().all():
        raise ValueError("Date/Time columns could not be parsed to datetime")
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
    # Collapse intraday to daily by last close per calendar day
    s = s.groupby(s.index.normalize()).last().sort_index().astype("float64")
    return s

def load_series(folder: str, sym: str) -> Optional[pd.Series]:
    p = Path(folder) / f"{sym}Raw.txt"
    try:
        return read_close_daily(p)
    except Exception as e:
        print(f"[WARN] {sym}: {e}")
        return None

# =========================
# Core: Week-of-Month rotation
# =========================
def compute_tdom_and_week(idx: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Compute TDOM (1..N per month) and WeekOfMonth (1.., clamp ≥4→4) for a given daily index."""
    y = idx.year
    m = idx.month
    # TDOM: 1.. within each Y-M
    df = pd.DataFrame({"Y": y, "M": m}, index=idx)
    tdom = df.groupby(["Y","M"]).cumcount().to_numpy() + 1
    # Week: 1 + floor((tdom-1)/5), then clamp >=4 → 4 (week4 and week5 same)
    week = ( (tdom - 1) // 5 ) + 1
    week = np.where(week >= 4, 4, week)
    return tdom.astype(int), week.astype(int)

def simulate_week_rotation(
    symbols_by_week: List[str],   # [w1, w2, w3, w4/5]
    closes: Dict[str, pd.Series],
    bench: Optional[pd.Series],
    start_equity: float = START_EQUITY,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Rotate symbols by trading week of month. Returns (equity, detail, bench_eq)."""
    if len(symbols_by_week) != 4:
        raise ValueError("symbols_by_week must have 4 tickers: [week1, week2, week3, week4/5]")

    needed_syms = list(dict.fromkeys(symbols_by_week))  # unique, preserve order
    # Intersection calendar across required symbols
    idx = None
    for s in needed_syms:
        ser = closes.get(s)
        if ser is None:
            raise RuntimeError(f"Missing data for {s}")
        idx = ser.index if idx is None else idx.intersection(ser.index)

    if idx is None or len(idx) < 200:
        raise RuntimeError("Not enough common dates across symbols.")
    idx = idx.sort_values()

    # Precompute returns per symbol on common calendar
    ret_map: Dict[str, np.ndarray] = {}
    for s in needed_syms:
        r = closes[s].reindex(idx).pct_change().fillna(0.0).to_numpy()
        ret_map[s] = np.maximum(r, -0.999999)  # avoid blow-ups

    # TDOM and Week
    tdom, week = compute_tdom_and_week(idx)

    # Build mapping week -> symbol
    week_to_sym = {1: symbols_by_week[0],
                   2: symbols_by_week[1],
                   3: symbols_by_week[2],
                   4: symbols_by_week[3]}  # week 4 & 5 mapped to index 4 (already clamped)

    applied = np.zeros(len(idx), dtype=float)
    held = np.empty(len(idx), dtype=object)
    for w in (1,2,3,4):
        sym = week_to_sym[w]
        mask = (week == w)
        if mask.any():
            # If sym wasn't in needed_syms (shouldn't happen), guard anyway
            if sym not in ret_map:
                raise RuntimeError(f"Symbol {sym} not loaded for week {w}")
            applied[mask] = ret_map[sym][mask]
            held[mask] = sym

    # Equity & detail
    eq_vals = (1.0 + applied).cumprod() * float(start_equity)
    eq = pd.Series(eq_vals, index=idx, name="Equity")
    detail = pd.DataFrame({
        "Y": idx.year,
        "M": idx.month,
        "TDOM": tdom,
        "Week": week,
        "Held": held,
        "AppliedRet": applied,
        "Equity": eq_vals,
    }, index=idx)

    # Benchmark on same calendar
    bench_eq = pd.Series(dtype="float64")
    if bench is not None:
        b = bench.reindex(idx).ffill()
        br = b.pct_change().fillna(0.0).to_numpy()
        br = np.maximum(br, -0.999999)
        bench_eq = pd.Series((1.0 + br).cumprod() * float(start_equity), index=idx, name="Benchmark_Equity")

    return eq, detail, bench_eq

# =========================
# Metrics & Plotting
# =========================
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

    fig = plt.figure(figsize=(12, 5)); ax = fig.add_subplot(111)
    if not bqp.empty and bqp.notna().sum() >= 2:
        ax.plot(bqp.index, bqp.values, label=f"{BENCHMARK} B&H", linestyle="--", linewidth=1.6, zorder=1)
    ax.plot(eqp.index, eqp.values, label=name, linewidth=1.8, zorder=2)
    ax.set_title(f"{name} (rebased @ {ps.date()})", fontsize=10)
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(outdir / f"{name}_equity_vs_{BENCHMARK}.png", dpi=130)
    plt.close(fig)

# =========================
# Runner
# =========================
def run_all():
    outp = Path(OUTDIR); outp.mkdir(parents=True, exist_ok=True)

    # Determine all symbols we need (strategies + benchmark)
    needed = {BENCHMARK}
    for _name, syms in STRATEGIES:
        needed.update(syms)

    # Load closes
    closes: Dict[str, pd.Series] = {}
    for sym in sorted(needed):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable data for {sym}")
        closes[sym] = ser.sort_index()

    bench = closes.get(BENCHMARK)

    rows = []
    for name, syms in STRATEGIES:
        try:
            eq, detail, bench_eq = simulate_week_rotation(syms, closes, bench, start_equity=START_EQUITY)
            # Save artifacts
            detail.to_csv(Path(OUTDIR) / f"{name}_detail.csv", index=True)
            eq.rename("Equity").to_csv(Path(OUTDIR) / f"{name}_equity.csv", header=True)
            plot_equity(name, eq, bench_eq, Path(OUTDIR), PLOT_START)
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

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run_all()
