#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monthly Gate Strategies using 200D/50D SMA (and SMA on spread) — NO LOOKAHEAD (IDLE-ready)

Decisions are made *at the previous month's last trading day* (prev_eom), using only
data available up to that date. Holdings are applied for all trading days in the next month.

Strategies:
  1) SMH,GLD : if SMH > SMA200(SMH) → hold SMH else GLD
  2) XLK,GLD : if XLK > SMA200(XLK) → hold XLK else GLD
  3) GLD,QQQ : if GLD < SMA200(GLD) → hold GLD else QQQ
  4) SPY,QQQ : if SPY < SMA200(SPY) → hold SPY else QQQ
  5) GLD,QQQ : spread(GLD,QQQ) vs SMA200(spread); if spread > SMA → GLD else QQQ
  6) QQQ,SPY : spread(QQQ,SPY) vs SMA50(spread);  if spread > SMA → QQQ else SPY
  7) XLE,XLK : spread(XLE,XLK) vs SMA50(spread);  if spread > SMA → XLE else XLK

Execution calendar is the intersection of the two symbols' daily indices per strategy.
Benchmark is SPY continuous buy & hold on the same pair calendar (for fair overlay).

Outputs (./ma_monthly_gates_outputs):
  • <name>_detail.csv  (Y, M, Pick, AppliedRet, Equity)
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
import itertools as it

# =========================
# User configuration
# =========================
DATA_FOLDER = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR = "./ma_monthly_gates_outputs"
PLOT_START = "2010-01-01"
START_EQUITY = 100_000.0
BENCHMARK = "SPY"
VERBOSE = True

# Spread choice: "logratio" (ln(A/B)) or "diff" (A - B)
SPREAD_TYPE = "logratio"

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
# Core engines
# =========================
def _pair_calendar(closeA: pd.Series, closeB: pd.Series) -> pd.DatetimeIndex:
    idx = closeA.index.intersection(closeB.index)
    if len(idx) < 120:
        raise RuntimeError("Pair calendar too short.")
    return idx.sort_values()

def _month_iter(idx: pd.DatetimeIndex):
    """Yield period months present in idx in chronological order."""
    ym = pd.PeriodIndex(idx, freq="M")
    for p in pd.Index(ym.unique()).sort_values():
        yield p

def simulate_monthly_gate_singleA(
    A: pd.Series,
    B: pd.Series,
    bench: Optional[pd.Series],
    ma_len: int,
    rule: str,  # "above" => pick A if A > MA; "below" => pick A if A < MA
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Decide with A's SMA(ma_len) at prev_eom (A-native SMA, no lookahead)."""
    idx = _pair_calendar(A, B)
    a_native = A.sort_index().astype("float64")
    sma = a_native.rolling(int(ma_len), min_periods=int(ma_len)).mean()

    A2 = A.reindex(idx).astype("float64")
    B2 = B.reindex(idx).astype("float64")
    rA = A2.pct_change()
    rB = B2.pct_change()

    ym = pd.PeriodIndex(idx, freq="M")
    months_arr = np.asarray(ym)
    strat_r = pd.Series(0.0, index=idx, dtype="float64")
    picked = pd.Series("", index=idx, dtype="object")

    # helper: last A-native value <= ts
    a_dates = a_native.index.values
    import bisect
    def last_at_or_before(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        pos = bisect.bisect_right(a_dates, ts.to_datetime64()) - 1
        return pd.Timestamp(a_dates[pos]) if pos >= 0 else None

    for p in pd.Index(ym.unique()).sort_values():
        prev_p = p - 1
        prev_mask = (months_arr == prev_p)
        if not prev_mask.any():  # first month or gap
            continue
        prev_eom = idx[np.where(prev_mask)[0][-1]]

        ref = last_at_or_before(prev_eom)
        if ref is None:
            continue
        a_val = float(a_native.get(ref, np.nan))
        ma_val = float(sma.get(ref, np.nan))
        if not (np.isfinite(a_val) and np.isfinite(ma_val)):
            continue

        use_A = (a_val > ma_val) if rule == "above" else (a_val < ma_val)

        curr_mask = (months_arr == p)
        rows = np.where(curr_mask)[0]
        if rows.size == 0:
            continue
        chosen = rA if use_A else rB
        pr = chosen.iloc[rows].astype(float).to_numpy()
        pr = np.maximum(pr, -0.999999)
        strat_r.iloc[rows] = pr
        picked.iloc[rows] = "A" if use_A else "B"

    eq = pd.Series((1.0 + strat_r).cumprod() * float(START_EQUITY), index=idx, name="Equity")

    bench_eq = pd.Series(dtype="float64")
    if bench is not None:
        b = bench.reindex(idx).ffill()
        br = b.pct_change().fillna(0.0).to_numpy()
        br = np.maximum(br, -0.999999)
        bench_eq = pd.Series((1.0 + br).cumprod() * float(START_EQUITY), index=idx, name="Benchmark_Equity")

    detail = pd.DataFrame({
        "Y": idx.year, "M": idx.month,
        "Pick": picked,
        "AppliedRet": strat_r,
        "Equity": eq,
    }, index=idx)

    return eq, detail, bench_eq

def simulate_monthly_gate_spread(
    A: pd.Series,
    B: pd.Series,
    bench: Optional[pd.Series],
    ma_len: int,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Spread gating at prev_eom on pair calendar: if spread > SMA → pick A; else pick B."""
    idx = _pair_calendar(A, B)
    A2 = A.reindex(idx).astype("float64")
    B2 = B.reindex(idx).astype("float64")
    rA = A2.pct_change()
    rB = B2.pct_change()

    # spread on pair calendar
    if SPREAD_TYPE.lower() == "logratio":
        spread = np.log(A2 / B2)
    else:
        spread = A2 - B2
    sma = spread.rolling(int(ma_len), min_periods=int(ma_len)).mean()

    ym = pd.PeriodIndex(idx, freq="M")
    months_arr = np.asarray(ym)
    strat_r = pd.Series(0.0, index=idx, dtype="float64")
    picked = pd.Series("", index=idx, dtype="object")

    for p in pd.Index(ym.unique()).sort_values():
        prev_p = p - 1
        prev_mask = (months_arr == prev_p)
        if not prev_mask.any():
            continue
        prev_eom = idx[np.where(prev_mask)[0][-1]]

        s_val = float(spread.get(prev_eom, np.nan))
        m_val = float(sma.get(prev_eom, np.nan))
        if not (np.isfinite(s_val) and np.isfinite(m_val)):
            continue

        use_A = (s_val > m_val)  # above SMA → long A; else B

        curr_mask = (months_arr == p)
        rows = np.where(curr_mask)[0]
        if rows.size == 0:
            continue
        chosen = rA if use_A else rB
        pr = chosen.iloc[rows].astype(float).to_numpy()
        pr = np.maximum(pr, -0.999999)
        strat_r.iloc[rows] = pr
        picked.iloc[rows] = "A" if use_A else "B"

    eq = pd.Series((1.0 + strat_r).cumprod() * float(START_EQUITY), index=idx, name="Equity")

    bench_eq = pd.Series(dtype="float64")
    if bench is not None:
        b = bench.reindex(idx).ffill()
        br = b.pct_change().fillna(0.0).to_numpy()
        br = np.maximum(br, -0.999999)
        bench_eq = pd.Series((1.0 + br).cumprod() * float(START_EQUITY), index=idx, name="Benchmark_Equity")

    detail = pd.DataFrame({
        "Y": idx.year, "M": idx.month,
        "Spread": spread, "SpreadMA": sma,
        "Pick": picked,
        "AppliedRet": strat_r,
        "Equity": eq,
    }, index=idx)

    return eq, detail, bench_eq

# =========================
# Strategy wiring
# =========================
STRATS = [
    # name, type, A, B, ma_len, rule
    # type: "singleA" (uses A's SMA), "spread" (uses spread(A,B))
    ("SMH_over200_else_GLD", "singleA", "SMH", "GLD", 200, "above"),
    ("XLK_over200_else_GLD", "singleA", "XLK", "GLD", 200, "above"),
    ("GLD_below200_else_QQQ", "singleA", "GLD", "QQQ", 200, "below"),
    ("SPY_below200_else_QQQ", "singleA", "SPY", "QQQ", 200, "below"),
    ("GLDQQQ_spread_over200", "spread",  "GLD", "QQQ", 200, "above"),  # rule fixed in spread engine (above)
    ("QQQSPY_spread_over50",  "spread",  "QQQ", "SPY",  50,  "above"),
    ("XLEXLK_spread_over50",  "spread",  "XLE", "XLK",  50,  "above"),
]

# =========================
# Runner
# =========================
def run_all():
    outp = Path(OUTDIR); outp.mkdir(parents=True, exist_ok=True)

    # collect needed symbols
    needed = {BENCHMARK}
    for name, typ, A, B, _ma, _rule in STRATS:
        needed.add(A); needed.add(B)

    # load closes
    closes: Dict[str, pd.Series] = {}
    for sym in sorted(needed):
        ser = load_series(DATA_FOLDER, sym)
        if ser is None:
            raise RuntimeError(f"Missing or unreadable data for {sym}")
        closes[sym] = ser.sort_index()

    bench = closes.get(BENCHMARK)

    rows = []
    for name, typ, A, B, ma_len, rule in STRATS:
        try:
            if typ == "singleA":
                eq, detail, bench_eq = simulate_monthly_gate_singleA(
                    closes[A], closes[B], bench, ma_len=ma_len, rule=rule
                )
            elif typ == "spread":
                eq, detail, bench_eq = simulate_monthly_gate_spread(
                    closes[A], closes[B], bench, ma_len=ma_len
                )
            else:
                raise ValueError(f"Unknown type: {typ}")

            # save artifacts
            detail.to_csv(outp / f"{name}_detail.csv", index=True)
            eq.rename("Equity").to_csv(outp / f"{name}_equity.csv", header=True)

            # plot
            plot_equity(name, eq, bench_eq, outp, PLOT_START)

            # metrics
            ms = metrics_from_equity(eq)
            rows.append({"Name": name, "A": A, "B": B, "Type": typ, "MA": ma_len,
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
