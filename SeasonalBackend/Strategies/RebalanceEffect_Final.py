#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rebalance Effect (No Lookahead) — FIXED ndarray .to_numpy error

For each month:
  • Compute MTD returns from TDOM1 close to TDOM{decision} close for A and B.
  • Pick the WORSE performer.
  • ENTER next trading day (after decision) and hold until EOM (~last ~7 TDOMs).
Strategy #5 uses decision TDOM=7; others use 15.

Equity is daily close→close on the pair's intersection calendar.
Benchmark (SPY) accrues ONLY on the strategy’s active days (flat otherwise).

Outputs (./rebalance_effect_outputs):
  • <name>_detail.csv    (Y, M, TDOM, Pick, AppliedRet, Equity, IsActive)
  • <name>_equity.csv
  • <name>_equity_vs_SPYActiveOnly.png
  • summary_metrics.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# =========================
# User configuration
# =========================
DATA_FOLDER = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR      = "./rebalance_effect_outputs"
BENCHMARK   = "SPY"
PLOT_START  = "2010-01-01"
START_EQUITY = 100_000.0
VERBOSE = True

# Strategies: (name, A, B, decision_tdom)
STRATEGIES: List[Tuple[str,str,str,int]] = [
    ("TLT_IWM_underperf_on15_hold_toEOM", "TLT", "IWM", 15),
    ("SPY_TLT_underperf_on15_hold_toEOM", "SPY", "TLT", 15),
    ("SMH_HYG_underperf_on15_hold_toEOM", "SMH", "HYG", 15),
    ("TLT_EEM_underperf_on15_hold_toEOM", "TLT", "EEM", 15),
    ("DIA_SMH_underperf_on7_hold_toEOM",  "DIA", "SMH",  7),
]

# =========================
# IO helpers (robust loader)
# =========================
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
        if VERBOSE: print(f"[WARN] {sym}: {e}")
        return None

# =========================
# Core calc
# =========================
def compute_tdom(idx: pd.DatetimeIndex) -> np.ndarray:
    """Trading day of month (1..N) along idx."""
    df = pd.DataFrame({"Y": idx.year, "M": idx.month}, index=idx)
    return (df.groupby(["Y","M"]).cumcount().to_numpy() + 1).astype(int)

def simulate_underperf_lastweek(
    closeA: pd.Series,
    closeB: pd.Series,
    bench: Optional[pd.Series],
    decision_td: int = 15,
    start_equity: float = START_EQUITY,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    At each month:
      - If month has at least 'decision_td' sessions on the pair calendar:
          • Compute MTD for A,B using first close of month → decision close.
          • Pick WORSE performer at decision close.
          • ENTER next day (decision_td+1) and HOLD to EOM.
      - Else: no position that month.
    """
    # Pair calendar (intersection)
    idx = closeA.index.intersection(closeB.index)
    if len(idx) < 200:
        raise RuntimeError("Pair calendar too short.")
    idx = idx.sort_values()

    A = closeA.reindex(idx).astype("float64")
    B = closeB.reindex(idx).astype("float64")
    rA = A.pct_change().fillna(0.0).to_numpy()
    rB = B.pct_change().fillna(0.0).to_numpy()
    rA = np.maximum(rA, -0.999999); rB = np.maximum(rB, -0.999999)

    tdom = compute_tdom(idx)
    months = idx.month
    years  = idx.year

    # First close of month for A/B (on pair calendar)
    month_key = idx.to_period("M")
    firstA = A.groupby(month_key).transform("first")
    firstB = B.groupby(month_key).transform("first")

    applied = np.zeros(len(idx), dtype=float)
    picked  = np.empty(len(idx), dtype=object)
    active  = np.zeros(len(idx), dtype=bool)

    # iterate months
    unique_months = pd.Index(pd.PeriodIndex(idx, freq="M").unique()).sort_values()
    for p in unique_months:
        m_mask = (month_key == p)                # boolean mask (np.ndarray on old pandas)
        if not np.asarray(m_mask).any():
            continue
        # decision day row (TDOM == decision_td)
        dec_mask = np.asarray(m_mask) & (tdom == decision_td)
        if not dec_mask.any():
            # not enough sessions to reach decision_td => skip month
            continue
        # decision at the end of decision day (close)
        dec_iloc = np.where(dec_mask)[0][-1]
        # MTD up to decision (no lookahead)
        A_mtd = float(A.iloc[dec_iloc] / firstA.iloc[dec_iloc] - 1.0)
        B_mtd = float(B.iloc[dec_iloc] / firstB.iloc[dec_iloc] - 1.0)
        # pick WORSE (lower MTD)
        use_A = (A_mtd <= B_mtd)  # <= keeps A when tie

        # hold from next day after decision to EOM
        after_dec_same_month = (np.arange(len(idx)) > dec_iloc) & np.asarray(m_mask)
        if after_dec_same_month.any():
            chosen = rA if use_A else rB
            applied[after_dec_same_month] = chosen[after_dec_same_month]
            picked[after_dec_same_month]  = "A" if use_A else "B"
            active[after_dec_same_month]  = True
        # decision day has zero return assigned (enter next day)

    # Equity
    eq_vals = (1.0 + applied).cumprod() * float(start_equity)
    eq = pd.Series(eq_vals, index=idx, name="Equity")

    # Benchmark: SPY accrues ONLY on active days
    bench_eq = pd.Series(dtype="float64")
    if bench is not None:
        b = bench.reindex(idx).ffill()
        br = b.pct_change().fillna(0.0).to_numpy()
        br = np.maximum(br, -0.999999)
        br_active = np.where(active, br, 0.0)
        bench_eq = pd.Series((1.0 + br_active).cumprod() * float(start_equity), index=idx, name="Benchmark_Equity")

    detail = pd.DataFrame({
        "Y": years, "M": months, "TDOM": tdom,
        "Pick": picked, "IsActive": active,
        "AppliedRet": applied, "Equity": eq_vals
    }, index=idx)

    return eq, detail, bench_eq

# =========================
# Metrics & plotting
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

    fig = plt.figure(figsize=(12,5)); ax = fig.add_subplot(111)
    if not bqp.empty and bqp.notna().sum() >= 2:
        ax.plot(bqp.index, bqp.values, label=f"{BENCHMARK} B&H (active days only)", linestyle="--", linewidth=1.6, zorder=1)
    ax.plot(eqp.index, eqp.values, label=name, linewidth=1.8, zorder=2)
    ax.set_title(f"{name} (rebased @ {ps.date()})", fontsize=10)
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(outdir / f"{name}_equity_vs_SPYActiveOnly.png", dpi=130)
    plt.close(fig)

# =========================
# Runner
# =========================
def run_all():
    outp = Path(OUTDIR); outp.mkdir(parents=True, exist_ok=True)

    # symbols we need
    needed = {BENCHMARK}
    for _, A, B, _d in STRATEGIES:
        needed.add(A); needed.add(B)

    # load
    closes: Dict[str, pd.Series] = {}
    for sym in sorted(needed):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable data for {sym}")
        closes[sym] = ser.sort_index()

    bench = closes.get(BENCHMARK)

    rows = []
    for name, A, B, decision_td in STRATEGIES:
        try:
            eq, detail, bench_eq = simulate_underperf_lastweek(
                closes[A], closes[B], bench, decision_td=decision_td, start_equity=START_EQUITY
            )
            # Save artifacts
            detail.to_csv(outp / f"{name}_detail.csv", index=True)
            eq.rename("Equity").to_csv(outp / f"{name}_equity.csv", header=True)
            # Plot
            plot_equity(name, eq, bench_eq, outp, PLOT_START)
            # Metrics
            ms = metrics_from_equity(eq)
            rows.append({"Name": name, "A": A, "B": B, "DecisionTD": decision_td,
                         "CAGR": ms["CAGR"], "TotalRet": ms["TotalRet"], "MaxDD": ms["MaxDD"]})
            if VERBOSE:
                print(f"[OK] {name}: CAGR={ms['CAGR']:.3%}  Total={ms['TotalRet']:.2%}  MaxDD={ms['MaxDD']:.2%}")
        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(outp / "summary_metrics.csv", index=False)
        print(f"\nDone. Outputs → {Path(OUTDIR).resolve()}")
    else:
        print("No strategies produced output. Check data coverage / symbol names.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run_all()
