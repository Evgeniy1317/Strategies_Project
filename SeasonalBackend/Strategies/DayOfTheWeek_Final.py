#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Day-of-Week Futures Strategies (Open→Open, 1-day hold)
======================================================

Data:
  - Folder: C:\Users\david\BuildAlpha\BuildAlpha\Data
  - Files: <SYMBOL>Raw.txt (e.g., ESRaw.txt, GCRaw.txt, etc.)
  - Needs at least 'Date', 'Open', and a close-like column ('Settle' preferred, else 'Close')

Mechanics:
  - Evaluate condition on day t's close; if True → enter next bar's OPEN (t+1), exit the following bar's OPEN (t+2).
  - Position size is constant = floor(ACCOUNT / margin) contracts.
  - PnL per trade (long):  (Open[t+2] - Open[t+1]) * point_value * contracts
             (short):      (Open[t+1] - Open[t+2]) * point_value * contracts
  - Equity starts at ACCOUNT and updates only on exit days.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ==========================
# Defaults
# ==========================
DATA_FOLDER = r"C:\Users\david\BuildAlpha\BuildAlpha\Data"
OUTDIR = "./dow_strats_outputs"
ACCOUNT_START = 100_000.0
PLOT_START_DEFAULT = "2010-01-01"   # plot 2010→present (metrics still full history)


# ==========================
# Flexible file reader
# ==========================
def read_futures_ohlc(path: Path) -> pd.DataFrame:
    """
    Read <SYMBOL>Raw.txt and return a DataFrame with index=Date, columns=['Open','Close'].
    Prefers 'Settle' for Close if present; else 'Close'.
    """
    df = None
    for sep in [None, ",", "\t", r"\s+"]:
        try:
            tmp = pd.read_csv(path, sep=sep, engine="python")
            if len(tmp):
                df = tmp
                break
        except Exception:
            pass
    if df is None or df.empty:
        raise ValueError(f"Could not read: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    # Date
    date_col = None
    for cand in ["date", "timestamp", "time"]:
        if cand in lower:
            date_col = lower[cand]; break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)

    # Open
    open_col = None
    for cand in ["open", "Open", "OPEN"]:
        if cand in df.columns or cand.lower() in lower:
            open_col = cand if cand in df.columns else lower[cand.lower()]
            break
    if open_col is None:
        raise ValueError(f"No 'Open' column found in {path}")

    # Close-like (prefer settle)
    close_col = None
    for cand in ["settle", "Settle", "SETTLE", "close", "Close", "CLOSE", "last"]:
        if cand in df.columns or cand.lower() in lower:
            close_col = cand if cand in df.columns else lower[cand.lower()]
            break
    if close_col is None:
        raise ValueError(f"No close/settle column found in {path}")

    out = pd.DataFrame({
        "Open": pd.to_numeric(df[open_col], errors="coerce"),
        "Close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    return out.dropna().astype("float64")


# ==========================
# Utilities & metrics
# ==========================
DOW_NAME = np.array(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

def metrics_from_equity(eq: pd.Series) -> Dict[str, float]:
    """Compute basic metrics from a daily equity curve (full history)."""
    eq = eq.dropna()
    if eq.empty or len(eq) < 5:
        return {"CAGR": np.nan, "TotalRet": np.nan, "MaxDD": np.nan, "Ulcer": np.nan}
    years = len(eq) / 252.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / max(1e-12, years)) - 1.0)
    roll = np.maximum.accumulate(eq.values)
    dd = eq.values / roll - 1.0
    maxdd = float(dd.min())
    ulcer = float(np.sqrt(np.mean(dd**2)))
    return {"CAGR": cagr, "TotalRet": total, "MaxDD": maxdd, "Ulcer": ulcer}

def ensure_outdir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


# ==========================
# Strategy definitions
# ==========================
# side: 'long' or 'short'
# cmp: 'le' (Close <= Open), 'gt' (Close > Open), 'any' (ignore)
STRATS = [
    {"name": "GC_Thu_down_long",     "symbol": "GC", "margin": 18700, "point": 100,   "side": "long",  "dow": "Thursday", "cmp": "le"},
    {"name": "ES_Mon_down_long",     "symbol": "ES", "margin": 23000, "point": 50,    "side": "long",  "dow": "Monday",   "cmp": "le"},

    # CHANGED: CL = SHORT when (Close > Open) AND Friday
    {"name": "CL_Fri_up_short",      "symbol": "CL", "margin": 6800,  "point": 1000,  "side": "short", "dow": "Friday",   "cmp": "gt"},

    {"name": "O_Tue_up_long",        "symbol": "O",  "margin": 1375,  "point": 50,    "side": "long",  "dow": "Tuesday",  "cmp": "gt"},

    # KC = LONG on Fridays (no body condition) — this was already correct
    {"name": "KC_Fri_long",          "symbol": "KC", "margin": 12800, "point": 375,   "side": "long",  "dow": "Friday",   "cmp": "any"},

    {"name": "RB_Fri_up_short",      "symbol": "RB", "margin": 7500,  "point": 42000, "side": "short", "dow": "Friday",   "cmp": "gt"},
    {"name": "VX_Mon_up_short",      "symbol": "VX", "margin": 15500, "point": 1000,  "side": "short", "dow": "Monday",   "cmp": "gt"},

    # From earlier tweak: NG is SHORT on Wednesdays
    {"name": "NG_Wed_short",         "symbol": "NG", "margin": 3800,  "point": 10000, "side": "short", "dow": "Wednesday","cmp": "any"},
]



# ==========================
# Signals & Backtester
# ==========================
def build_signals(df: pd.DataFrame, dow: str, cmp: str) -> pd.Series:
    """Entry-condition True on bar t if DOW matches and body rule holds."""
    idx = df.index
    dow_match = pd.Series(DOW_NAME[idx.dayofweek] == dow, index=idx)
    if cmp == "le":
        body = df["Close"] <= df["Open"]
    elif cmp == "gt":
        body = df["Close"] > df["Open"]
    else:
        body = pd.Series(True, index=idx)
    return (dow_match & body)

def backtest_one(symbol: str, margin: float, point_value: float, side: str,
                 df: pd.DataFrame, name: str, account_start: float) -> Dict:
    """
    Signal on t → enter at Open[t+1], exit at Open[t+2]; constant contracts = floor(account/margin).
    """
    contracts = int(np.floor(account_start / float(margin)))
    if contracts <= 0:
        raise ValueError(f"{name}: contracts computed as 0 (margin={margin})")

    opens = df["Open"].values
    idx = df.index
    sig = df["_signal"].astype(bool).values
    sig_idx = np.where(sig)[0]

    entries, exits, pl = [], [], []
    for t in sig_idx:
        e = t + 1
        x = t + 2
        if x >= len(df) or e >= len(df):
            continue
        open_e, open_x = opens[e], opens[x]
        if side == "long":
            trade_pl = (open_x - open_e) * point_value * contracts
        else:
            trade_pl = (open_e - open_x) * point_value * contracts
        entries.append(idx[e]); exits.append(idx[x]); pl.append(trade_pl)

    trades = pd.DataFrame({"Entry": entries, "Exit": exits, "PnL": pl})
    if not trades.empty:
        trades.index = trades["Exit"]  # anchor to exit day

    eq = pd.Series(account_start, index=idx, dtype="float64")
    pnl_series = trades["PnL"].groupby(trades.index).sum().reindex(idx).fillna(0.0) if not trades.empty else pd.Series(0.0, index=idx)
    eq = (eq + pnl_series.cumsum()).astype("float64")

    metr = metrics_from_equity(eq)
    return {
        "name": name, "symbol": symbol, "contracts": contracts,
        "point_value": point_value, "margin": margin,
        "equity": eq, "trades": trades, "metrics": metr,
    }


# ==========================
# Orchestrator
# ==========================
def run_all(data_dir: str, outdir: str, account: float, subset: List[str] | None, plot_start: str):
    outp = ensure_outdir(outdir)

    # Load required symbols
    needed = sorted(set(s["symbol"] for s in STRATS if (subset is None or s["symbol"] in subset)))
    data: Dict[str, pd.DataFrame] = {}
    for sym in needed:
        path = Path(data_dir) / f"{sym}Raw.txt"
        df = read_futures_ohlc(path)
        df["_DOW"] = DOW_NAME[df.index.dayofweek]
        data[sym] = df

    results = []
    for spec in STRATS:
        if subset is not None and spec["symbol"] not in subset:
            continue
        sym = spec["symbol"]; df = data[sym].copy()
        df["_signal"] = build_signals(df, spec["dow"], spec["cmp"])
        res = backtest_one(sym, spec["margin"], spec["point"], spec["side"], df, spec["name"], account)
        results.append(res)

        # Save trades and equity (full history)
        res["trades"].to_csv(outp / f"{spec['name']}_trades.csv", index=False)
        res["equity"].rename("Equity").to_csv(outp / f"{spec['name']}_equity.csv", header=True)

                # Plot: 2010-present (or user-specified start) — REBASED ON P&L SINCE plot_start
        try:
            import matplotlib.pyplot as plt
            start_dt = pd.Timestamp(plot_start)

            # Build a PnL-only equity from plot_start: 100k + cumulative PnL of trades whose Exit >= start_dt
            idx = res["equity"].index
            idx_slice = idx[idx >= start_dt]

            if len(idx_slice) < 2:
                print(f"[WARN] {spec['name']}: not enough data after {plot_start}; skipping plot.")
            else:
                # Sum PnL by exit date, restrict to window, and forward-fill on the date index
                trades = res["trades"]
                if not trades.empty:
                    pnl_by_day = trades.loc[trades["Exit"] >= start_dt, "PnL"] \
                                     .groupby(trades.loc[trades["Exit"] >= start_dt, "Exit"]).sum()
                    pnl_cum = pnl_by_day.reindex(idx_slice).fillna(0.0).cumsum()
                else:
                    pnl_cum = pd.Series(0.0, index=idx_slice)

                eq_plot = (ACCOUNT_START + pnl_cum).astype("float64")

                # Plot normalized to 1.0 at plot_start for clarity
                norm = eq_plot / float(eq_plot.iloc[0])

                fig = plt.figure(figsize=(10, 5)); ax = fig.add_subplot(111)
                ax.plot(norm.index, norm.values, label=spec["name"])
                ax.set_title(f"{spec['name']} — Equity (rebased to {plot_start} P&L)")
                ax.grid(True); ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(outp / f"{spec['name']}_equity.png", dpi=130)
                plt.close(fig)
        except Exception as e:
            print(f"[WARN] Plot failed for {spec['name']}: {e}")


    # Summary (metrics based on full history)
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({
            "Strategy": r["name"], "Symbol": r["symbol"], "Contracts": r["contracts"],
            "CAGR": m["CAGR"], "TotalRet": m["TotalRet"], "MaxDD": m["MaxDD"], "Ulcer": m["Ulcer"],
        })
    summary = pd.DataFrame(rows).sort_values(["CAGR","TotalRet"], ascending=False)
    summary.to_csv(outp / "summary_dow_strategies.csv", index=False)

    with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
        print("\n=== Day-of-Week Strategies Summary (full-history metrics) ===")
        print(summary.to_string(index=False))
        print(f"\nOutputs written to: {outp.resolve()}")


# ==========================
# CLI
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DATA_FOLDER, help="Data folder with <SYMBOL>Raw.txt files")
    ap.add_argument("--outdir", type=str, default=OUTDIR, help="Output folder for PNG/CSV")
    ap.add_argument("--account", type=float, default=ACCOUNT_START, help="Starting account value")
    ap.add_argument("--subset", nargs="*", default=None, help="Optional list of symbols to run (e.g., --subset ES CL)")
    ap.add_argument("--plot-start", type=str, default=PLOT_START_DEFAULT, help="Plot start date (YYYY-MM-DD), default 2010-01-01")
    args = ap.parse_args()

    run_all(args.data, args.outdir, args.account, subset=args.subset, plot_start=args.plot_start)


if __name__ == "__main__":
    main()
