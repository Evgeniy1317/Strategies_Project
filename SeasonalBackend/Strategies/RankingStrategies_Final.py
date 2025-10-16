#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ranking Seasonality — Selected Strategy Updates (NO LOOKAHEAD)

For each strategy (subset of symbols):
  • For each calendar month on the subset's intersection calendar:
      - Compute each symbol's average close→close return for the SAME calendar month
        over all prior completed years (lookback = 'to date', strictly < current year).
      - Pick per spec:
          - "worst":  buy worst-ranked (lowest avg)
          - "best":   buy best-ranked (highest avg)
          - "worst2": buy two worst equally (50/50)
      - Hold the pick(s) for the entire month on the subset calendar.
  • Plot equity curve vs SPY buy & hold (continuous) and write outputs.

Outputs (./ranking_selected_outputs):
  • <tag>_detail.csv   (daily detail)
  • <tag>_equity.csv
  • <tag>_equity_vs_SPY.png
  • summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# User config
# =========================
DATA_FOLDER   = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR        = "./ranking_selected_outputs"
PLOT_START    = "2010-01-01"
START_EQUITY  = 100_000.0
BENCHMARK     = "SPY"
VERBOSE       = True

# --- Strategies you asked for ---
# mode: "worst", "best", or "worst2"
STRATEGIES = [
    ("GDX,SMH,GLD,XLE,XHB",                "worst"),
    ("SMH,XLV,XLE,XHB",                    "worst"),
    ("QQQ,GDX,XLK,XHB",                    "worst"),
    ("GDX,GLD,XHB,SPY",                    "worst"),
    ("QQQ,GDX,XHB",                        "worst"),
    ("XLV,XLK,XHB",                        "worst"),
    ("SMH,GLD,EWG",                        "best"),
    ("QQQ,GDX,SMH,XLV,XHB",                "worst2"),
    ("GDX,SMH,XLV,XLE,XLP,XHB",            "worst2"),
]

# Rank requires a minimum number of past occurrences for that month to be eligible:
MIN_YEARS_REQ = 5   # adjust if needed

# =========================
# IO helpers
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

def read_close_daily_unique(path: Path) -> pd.Series:
    df = None
    for sep in [None, ",", "\t", r"\s+"]:
        try:
            tmp = pd.read_csv(path, sep=sep, engine="python")
            if len(tmp):
                df = tmp; break
        except Exception:
            pass
    if df is None or df.empty:
        raise ValueError(f"Could not read: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    idx = _coerce_datetime(df).normalize()
    df.index = idx
    lower = {c.lower(): c for c in df.columns}
    close_col = None
    for cand in ("close","settle","last","adj close","adj_close","adjusted close"):
        if cand in lower:
            close_col = lower[cand]
            break
    if close_col is None:
        for cand in ("Close","Settle","Last","Adj Close","AdjClose"):
            if cand in df.columns:
                close_col = cand; break
    if close_col is None:
        raise ValueError(f"No close-like column in {path}")

    s = pd.to_numeric(df[close_col], errors="coerce").dropna()
    s = s.groupby(s.index).last().sort_index().astype("float64")
    s = s[~s.index.duplicated(keep="last")]
    return s

def load_series(folder: str, sym: str) -> Optional[pd.Series]:
    p = Path(folder) / f"{sym}Raw.txt"
    try:
        return read_close_daily_unique(p)
    except Exception as e:
        if VERBOSE: print(f"[WARN] {sym}: {e}")
        return None

# =========================
# Calculations
# =========================
def monthly_returns_by_symbol(close: pd.Series) -> pd.DataFrame:
    """
    Monthly close→close returns. IMPORTANT: use the *returns* index for the frame
    to avoid off-by-one mismatches (first diff is NaN).
    """
    s = close.sort_index()
    month_last = s.resample("ME").last()
    mret = month_last.pct_change()           # same index as month_last
    mret = mret.iloc[1:]                     # drop the first (NaN) explicitly
    per = mret.index.to_period("M")          # USE mret.index, NOT month_last.index

    out = pd.DataFrame({"ret": mret.values}, index=per)
    out["year"] = per.year
    out["month"] = per.month

    # Safety check
    assert len(out) == len(mret), "monthly_returns_by_symbol: index/value length mismatch"
    return out

def intersection_calendar(symbols: List[str], closes: Dict[str, pd.Series]) -> pd.DatetimeIndex:
    idx = None
    for s in symbols:
        ser = closes[s]
        ser = ser[~ser.index.duplicated(keep="last")].sort_index()
        idx = ser.index if idx is None else idx.intersection(ser.index)
    if idx is None or len(idx) < 120:  # require reasonable overlap
        raise RuntimeError("Intersection calendar too short across group.")
    return idx.sort_values()

# =========================
# Simulator
# =========================
def simulate_rank_pick(
    symbols: List[str],
    mode: str,          # "worst" | "best" | "worst2"
    closes: Dict[str, pd.Series],
    bench: Optional[pd.Series],
    start_equity: float = START_EQUITY,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    # Execution calendar
    idx = intersection_calendar(symbols, closes)

    # Numeric month-id for robust slicing
    years  = idx.year.to_numpy()
    months = idx.month.to_numpy()
    mid    = years * 12 + months
    unique_mid = np.unique(mid)

    # Precompute histories and daily returns on exec calendar
    mrets: Dict[str, pd.DataFrame] = {s: monthly_returns_by_symbol(closes[s]) for s in symbols}
    daily_ret: Dict[str, np.ndarray] = {}
    for s in symbols:
        rr = closes[s].reindex(idx).pct_change().fillna(0.0).to_numpy()
        daily_ret[s] = np.maximum(rr, -0.999999)

    n = len(idx)
    applied = np.zeros(n, dtype=float)
    pick_lab = np.full(n, "", dtype=object)  # "SYM" or "SYM1|SYM2" for worst2

    for cur_mid in unique_mid:
        y = cur_mid // 12
        m = cur_mid % 12
        if m == 0:  # guard (shouldn't happen)
            y -= 1; m = 12

        # "to date" lookback = all prior completed years
        years_lb = list(range(int(y) - 1000, int(y)))

        # Build scores for this calendar month
        scores = []
        for s in symbols:
            df = mrets[s]
            sel = df[(df["month"] == m) & (df["year"].isin(years_lb))]
            if len(sel) >= MIN_YEARS_REQ:
                scores.append((s, float(sel["ret"].mean())))
        if not scores:
            continue

        # Ascending: worst..best
        scores.sort(key=lambda x: x[1])

        rows = np.flatnonzero(mid == cur_mid)
        if rows.size == 0:
            continue

        if mode == "worst":
            sym = scores[0][0]
            applied[rows] = daily_ret[sym][rows]
            pick_lab[rows] = sym

        elif mode == "best":
            sym = scores[-1][0]
            applied[rows] = daily_ret[sym][rows]
            pick_lab[rows] = sym

        elif mode == "worst2":
            if len(scores) < 2:
                continue
            s1, s2 = scores[0][0], scores[1][0]
            basket = 0.5 * daily_ret[s1][rows] + 0.5 * daily_ret[s2][rows]
            applied[rows] = basket
            pick_lab[rows] = f"{s1}|{s2}"

        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Equity & detail
    eq_vals = (1.0 + applied).cumprod() * float(start_equity)
    eq = pd.Series(eq_vals, index=idx, name="Equity")

    detail = pd.DataFrame(index=idx)
    detail["Y"] = years
    detail["M"] = months
    detail["Pick"] = pick_lab
    detail["AppliedRet"] = applied
    detail["Equity"] = eq_vals

    # Benchmark (continuous) on same calendar
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
        return {"CAGR": np.nan, "TotalRet": np.nan, "MaxDD": np.nan, "Sharpe": np.nan}
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((eq.index[-1] - eq.index[0]).days, 1) / 365.25
    cagr = float(np.exp(np.log(eq.iloc[-1] / eq.iloc[0]) / years) - 1.0) if eq.iloc[0] > 0 and years > 0 else np.nan
    peak = np.maximum.accumulate(eq.to_numpy())
    dd = eq.to_numpy() / peak - 1.0
    rets = eq.pct_change().dropna().to_numpy()
    ann_vol = rets.std(ddof=1) * np.sqrt(252.0) if rets.size else np.nan
    sharpe = ((eq.iloc[-1]/eq.iloc[0])**(1/years)-1)/ann_vol if (ann_vol and ann_vol>0 and years>0) else np.nan
    return {"CAGR": cagr, "TotalRet": total, "MaxDD": float(dd.min()), "Sharpe": sharpe}

def plot_equity(name: str, eq: pd.Series, bench: pd.Series, outdir: Path, plot_start: str):
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
        ax.plot(bqp.index, bqp.values, label=f"{BENCHMARK} B&H", linestyle="--", linewidth=1.6)
    ax.plot(eqp.index, eqp.values, label=name, linewidth=1.8)
    ax.set_title(f"{name} (rebased @ {ps.date()})", fontsize=10)
    ax.grid(True); ax.legend(loc="best"); fig.tight_layout()
    fig.savefig(outdir / f"{name}_equity_vs_{BENCHMARK}.png", dpi=130)
    plt.close(fig)

# =========================
# Runner
# =========================
def run_all():
    outp = Path(OUTDIR); outp.mkdir(parents=True, exist_ok=True)

    # Symbols needed: union of all sets + benchmark
    needed_syms = set()
    for s, _mode in STRATEGIES:
        needed_syms.update(map(str.strip, s.split(",")))
    needed_syms.add(BENCHMARK)

    # Load once
    closes: Dict[str, pd.Series] = {}
    for sym in sorted(needed_syms):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable data for {sym}")
        closes[sym] = ser.sort_index()
    bench = closes.get(BENCHMARK)

    # Run each strategy
    rows = []
    for i, (symlist, mode) in enumerate(STRATEGIES, start=1):
        syms = [s.strip() for s in symlist.split(",")]
        tag = f"{i:02d}_{mode.upper()}_" + "_".join(syms)
        try:
            eq, detail, bench_eq = simulate_rank_pick(syms, mode, closes, bench, START_EQUITY)
            # Plot & save
            plot_equity(tag, eq, bench_eq, outp, PLOT_START)
            detail.to_csv(outp / f"{tag}_detail.csv", index=True)
            eq.rename("Equity").to_csv(outp / f"{tag}_equity.csv", header=True)
            # Metrics
            m = metrics_from_equity(eq)
            rows.append({
                "Tag": tag,
                "Subset": ",".join(syms),
                "Mode": mode,
                "CAGR": m["CAGR"], "TotalRet": m["TotalRet"],
                "MaxDD": m["MaxDD"], "Sharpe": m["Sharpe"]
            })
            print(f"[OK] {tag}: CAGR={m['CAGR']:.3%}  Total={m['TotalRet']:.2%}  MaxDD={m['MaxDD']:.2%}")
        except Exception as e:
            print(f"[SKIP] {tag}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(outp / "summary.csv", index=False)
        print(f"\nSaved outputs → {outp.resolve()}")
    else:
        print("No strategies produced output. Check data coverage and names.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    run_all()
