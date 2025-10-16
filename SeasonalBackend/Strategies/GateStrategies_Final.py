#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gate Strategies (No Lookahead) — IDLE ready

Rule per strategy (A primary, B secondary):
  • Gate window (either a month or Q1=Jan–Mar):
      - Always hold A during the gate window.
  • At the gate window's END CLOSE, compute A's return over that gate window.
      - If gate_return > 0 → hold A from the next trading day through the day
        before the next gate starts.
      - Else → hold B for that interval.
  • Repeat each year. No lookahead: decision uses only data within the gate.

Execution calendar: A∩B daily dates. Benchmark: SPY continuous B&H on same calendar.

Outputs (./gate_outputs):
  • <name>_detail.csv (Y, M, Phase, AppliedRet, Equity)
  • <name>_equity.csv
  • <name>_equity_vs_SPY.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import itertools as it

# -----------------------
# User settings
# -----------------------
DATA_FOLDER = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR = "./gate_outputs"
PLOT_START = "2010-01-01"
BENCHMARK = "SPY"
START_EQUITY = 100_000.0
VERBOSE = True

# Strategies you asked for
# kind: "month" with month=1..12, or "q1" for Jan–Mar
STRATEGIES = [
    {"name": "SMH_vs_SLV_FEB",   "A": "SMH", "B": "SLV", "kind": "month", "month": 2},
    {"name": "SMH_vs_GLD_Q1",    "A": "SMH", "B": "GLD", "kind": "q1"},
    {"name": "GLD_vs_QQQ_OCT",   "A": "GLD", "B": "QQQ", "kind": "month", "month": 10},
    {"name": "SMH_vs_JNK_NOV",   "A": "SMH", "B": "JNK", "kind": "month", "month": 11},
    {"name": "GLD_vs_SMH_OCT",   "A": "GLD", "B": "SMH", "kind": "month", "month": 10},
]

# -----------------------
# IO helpers
# -----------------------
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
                df = t
                break
        except Exception:
            pass
    if df is None or df.empty:
        raise ValueError(f"Could not read: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    idx = _coerce_datetime(df)
    keep = ~idx.isna()
    df = df.loc[keep].copy()
    idx = idx[keep]
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
        if VERBOSE: print(f"[WARN] {sym}: {e}")
        return None

# -----------------------
# Core gate simulator
# -----------------------
def simulate_gate_strategy(
    closeA: pd.Series,
    closeB: pd.Series,
    bench: Optional[pd.Series],
    kind: str,
    month: Optional[int] = None,
    plot_start: str = PLOT_START,
    start_equity: float = START_EQUITY,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Build daily equity on A∩B calendar. For each cycle:
      • Gate window: hold A.
      • Decision at gate end (A gate return > 0 ? A : B) for the post-gate interval
        until the next gate starts.
    kind: "month" (requires month=1..12) or "q1".
    """
    # Execution calendar
    idx = closeA.index.intersection(closeB.index)
    if len(idx) < 200:
        raise RuntimeError("Pair calendar too short.")

    A = closeA.reindex(idx)
    B = closeB.reindex(idx)
    rA = A.pct_change()
    rB = B.pct_change()

    # Build monthly/quarter tags on pair calendar
    meta = pd.DataFrame(index=idx)
    meta["Y"] = idx.year
    meta["M"] = idx.month
    meta["Q"] = ((meta["M"] - 1) // 3 + 1)

    # Identify gate masks per cycle (by year)
    gate_spans = []  # list of (start_loc, end_loc, year_label)
    if kind == "month":
        if month is None or not (1 <= int(month) <= 12):
            raise ValueError("month must be 1..12 for kind='month'")
        for y in sorted(meta["Y"].unique()):
            mask = (meta["Y"] == y) & (meta["M"] == int(month))
            if mask.any():
                ilocs = np.where(mask.values)[0]
                gate_spans.append((ilocs[0], ilocs[-1], y))
    elif kind == "q1":
        # Jan–Mar of each year
        for y in sorted(meta["Y"].unique()):
            mask = (meta["Y"] == y) & (meta["M"].isin([1,2,3]))
            if mask.any():
                ilocs = np.where(mask.values)[0]
                gate_spans.append((ilocs[0], ilocs[-1], y))
    else:
        raise ValueError("Unsupported kind: use 'month' or 'q1'.")

    if not gate_spans:
        raise RuntimeError("No gate windows found on pair calendar.")

    strat_r = pd.Series(0.0, index=idx, dtype="float64")
    phase = pd.Series("", index=idx, dtype="object")

    # Iterate cycles in chronological order
    for k, (g_start, g_end, y) in enumerate(gate_spans):
        # 1) Gate window: apply A returns within [g_start .. g_end]
        gate_rows = np.arange(g_start, g_end + 1, dtype=int)
        gr = rA.iloc[gate_rows].to_numpy()
        gr = np.maximum(gr, -0.999999)
        strat_r.iloc[gate_rows] = gr
        phase.iloc[gate_rows] = "GATE(A)"

        # Gate return: product over gate_rows on A
        # (1+r).cumprod() over the segment; entry at first row, exit at last row close.
        gate_ret = float(np.prod(1.0 + rA.iloc[gate_rows].fillna(0.0).to_numpy()) - 1.0)

        # 2) Post-gate interval: from day after g_end through the day before next gate start
        if k + 1 < len(gate_spans):
            next_start = gate_spans[k + 1][0]
            hold_rows = np.arange(g_end + 1, next_start, dtype=int)
        else:
            # until end of available index
            hold_rows = np.arange(g_end + 1, len(idx), dtype=int)

        if hold_rows.size > 0:
            use_A = (gate_ret > 0.0)
            chosen = rA if use_A else rB
            pr = chosen.iloc[hold_rows].to_numpy()
            pr = np.maximum(pr, -0.999999)
            strat_r.iloc[hold_rows] = pr
            phase.iloc[hold_rows] = "HOLD(A)" if use_A else "HOLD(B)"

    # Equity
    eq = pd.Series((1.0 + strat_r).cumprod() * float(start_equity), index=idx, name="Equity")

    # Benchmark on same calendar (continuous B&H)
    bench_eq = pd.Series(dtype="float64")
    if bench is not None:
        b = bench.reindex(idx).ffill()
        br = b.pct_change().fillna(0.0).to_numpy()
        br = np.maximum(br, -0.999999)
        bench_eq = pd.Series((1.0 + br).cumprod() * float(start_equity), index=idx, name="Benchmark_Equity")

    detail = pd.DataFrame({
        "Y": meta["Y"],
        "M": meta["M"],
        "Phase": phase,
        "AppliedRet": strat_r,
        "Equity": eq,
    }, index=idx)

    return eq, detail, bench_eq

# -----------------------
# Metrics & plotting
# -----------------------
def metrics_from_equity(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 50:
        return {"CAGR": np.nan, "TotalRet": np.nan, "MaxDD": np.nan}
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days, 1) / 365.25
    if eq.iloc[0] <= 0 or eq.iloc[-1] <= 0 or years <= 0:
        cagr = np.nan
    else:
        cagr = float(np.exp(np.log(eq.iloc[-1] / eq.iloc[0]) / years) - 1.0)
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
        if len(bqp):
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

# -----------------------
# Runner
# -----------------------
def run_all():
    outp = Path(OUTDIR); outp.mkdir(parents=True, exist_ok=True)

    # Load all symbols we need
    needed = set([BENCHMARK])
    for s in STRATEGIES:
        needed.add(s["A"]); needed.add(s["B"])
    close_map: Dict[str, pd.Series] = {}
    for sym in sorted(needed):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable file for {sym}")
        close_map[sym] = ser.sort_index()

    # Loop strategies
    rows = []
    for s in STRATEGIES:
        name = s["name"]; A = s["A"]; B = s["B"]; kind = s["kind"]; month = s.get("month", None)
        try:
            eq, detail, bench = simulate_gate_strategy(
                close_map[A], close_map[B], close_map.get(BENCHMARK),
                kind=kind, month=month, plot_start=PLOT_START, start_equity=START_EQUITY
            )
            # Save artifacts
            detail.to_csv(Path(OUTDIR) / f"{name}_detail.csv", index=True)
            eq.rename("Equity").to_csv(Path(OUTDIR) / f"{name}_equity.csv", header=True)
            # Plot
            plot_equity(name, eq, bench, Path(OUTDIR), PLOT_START)
            # Metrics
            ms = metrics_from_equity(eq)
            rows.append({"Name": name, "A": A, "B": B, "Kind": kind, "Month": month,
                         "CAGR": ms["CAGR"], "TotalRet": ms["TotalRet"], "MaxDD": ms["MaxDD"]})
            if VERBOSE:
                print(f"[OK] {name}: CAGR={ms['CAGR']:.3%}  Total={ms['TotalRet']:.2%}  MaxDD={ms['MaxDD']:.2%}")
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(Path(OUTDIR) / "summary_metrics.csv", index=False)
        print(f"\nDone. Outputs → {Path(OUTDIR).resolve()}")
    else:
        print("No strategies produced output. Check data coverage / symbol names.")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run_all()
