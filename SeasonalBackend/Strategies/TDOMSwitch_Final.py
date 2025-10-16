#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Half-Month Rotation by TDOM — NO LOOKAHEAD (IDLE-ready)

Rule (per strategy):
  • Compute Trading Day of Month (TDOM) = 1..N for each calendar month on the pair's
    intersection calendar.
  • Hold symbol A for TDOM 1..K, and symbol B for TDOM (K+1)..EOM.
  • Fully invested, daily close→close returns on the pair calendar.
  • Benchmark: SPY continuous B&H on the same pair calendar (for fair overlay).

Outputs (./half_month_outputs):
  • <name>_detail.csv  (Y, M, TDOM, Held, AppliedRet, Equity)
  • <name>_equity.csv
  • <name>_equity_vs_SPY.png
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
OUTDIR = "./half_month_outputs"
PLOT_START = "2010-01-01"
START_EQUITY = 100_000.0
BENCHMARK = "SPY"
VERBOSE = True

# Strategies: (name, A, B, K)
# Hold A on TDOM 1..K, hold B on TDOM K+1..EOM
STRATEGIES: List[Tuple[str, str, str, int]] = [
    ("SMH_TLT_1to12__13toEOM", "SMH", "TLT", 12),
    ("QQQ_TLT_1to12__13toEOM", "QQQ", "TLT", 12),
    ("GLD_SMH_1to8__9toEOM",   "GLD", "SMH",  8),
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
# Core engine
# =========================
def compute_tdom(idx: pd.DatetimeIndex) -> np.ndarray:
    """Trading Day of Month (1..N) for each date in idx."""
    df = pd.DataFrame({"Y": idx.year, "M": idx.month}, index=idx)
    return (df.groupby(["Y","M"]).cumcount().to_numpy() + 1).astype(int)

def simulate_half_month(
    A: pd.Series,
    B: pd.Series,
    bench: Optional[pd.Series],
    k_first: int,                      # A holds on TDOM 1..k_first; B on k_first+1..EOM
    start_equity: float = START_EQUITY
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Return (equity, detail, bench_eq) on the intersection calendar (close→close)."""
    # Intersection calendar
    idx = A.index.intersection(B.index)
    if len(idx) < 200:
        raise RuntimeError("Pair calendar too short.")
    idx = idx.sort_values()

    A2 = A.reindex(idx).astype("float64")
    B2 = B.reindex(idx).astype("float64")
    rA = A2.pct_change().fillna(0.0).to_numpy()
    rB = B2.pct_change().fillna(0.0).to_numpy()
    rA = np.maximum(rA, -0.999999); rB = np.maximum(rB, -0.999999)

    # TDOM and masks
    tdom = compute_tdom(idx)
    months = idx.month
    years  = idx.year

    applied = np.zeros(len(idx), dtype=float)
    held = np.empty(len(idx), dtype=object)

    # first-half mask: 1..k_first  (applies per month)
    # vectorized: apply A where tdom <= k_first, else B
    mask_A = (tdom <= int(k_first))
    mask_B = ~mask_A

    applied[mask_A] = rA[mask_A]
    held[mask_A] = "A"
    applied[mask_B] = rB[mask_B]
    held[mask_B] = "B"

    eq_vals = (1.0 + applied).cumprod() * float(start_equity)
    eq = pd.Series(eq_vals, index=idx, name="Equity")

    detail = pd.DataFrame({
        "Y": years, "M": months, "TDOM": tdom,
        "Held": held, "AppliedRet": applied, "Equity": eq_vals
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

    # Collect needed symbols (strategies + benchmark)
    needed = {BENCHMARK}
    for _, a, b, _k in STRATEGIES:
        needed.add(a); needed.add(b)

    # Load closes
    closes: Dict[str, pd.Series] = {}
    for sym in sorted(needed):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable data for {sym}")
        closes[sym] = ser.sort_index()

    bench = closes.get(BENCHMARK)

    rows = []
    for name, A, B, k in STRATEGIES:
        try:
            eq, detail, bench_eq = simulate_half_month(
                closes[A], closes[B], bench, k_first=k, start_equity=START_EQUITY
            )
            # Save
            detail.to_csv(Path(OUTDIR) / f"{name}_detail.csv", index=True)
            eq.rename("Equity").to_csv(Path(OUTDIR) / f"{name}_equity.csv", header=True)
            plot_equity(name, eq, bench_eq, Path(OUTDIR), PLOT_START)

            ms = metrics_from_equity(eq)
            rows.append({"Name": name, "A": A, "B": B, "K": k,
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

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run_all()
