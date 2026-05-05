#!/usr/bin/env python3
"""
InvestorOz -- Parameter Optimizer
Fetches 500-stock universe, scores candidates as of SIM_START,
then grid-searches trading parameters to find combos that beat SPY.

Grid dimensions:
  n_stocks      : number of top-scored stocks to trade
  stop_loss     : maximum loss per position before exit
  take_profit   : target gain per position
  trailing_stop : exit if price falls this % from rolling peak

Data is downloaded ONCE; only the backtest loop is repeated per combo.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import date, timedelta
from itertools import product

# -------- CONFIG --------
INITIAL_CAPITAL = 10_000
HIST_MIN_PRICE  = 5.0     # min price as of DATA_START (historical, not today)
HIST_MIN_VOL    = 500_000 # min avg daily volume during warmup window (historical)

SIM_START  = date(2025, 1, 2)    # first trading day of 2025
SIM_END    = date(2025, 12, 31)  # last trading day of 2025
DATA_START = date(2024, 5, 1)    # warmup window for indicators

# Grid search space
GRID = {
    "n_stocks":      [3, 5, 7, 10],
    "stop_loss":     [0.05, 0.07, 0.10, 0.12, 0.15],
    "take_profit":   [0.15, 0.20, 0.25, 0.35, 0.50],
    "trailing_stop": [0.04, 0.06, 0.08, 0.10, 0.15],
}


# -------- HELPERS --------

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def price_on(df, d):
    rows = df[df.index.date <= d]
    return float(rows["Close"].iloc[-1]) if not rows.empty else None


def entry_on(df, d):
    rows = df[df.index.date >= d]
    if rows.empty:
        return None, None
    col = "Open" if "Open" in rows.columns else "Close"
    return float(rows[col].iloc[0]), rows.index[0].date()


# -------- UNIVERSE + DATA --------

def get_universe():
    print("Fetching universe from Wikipedia (S&P 500 + S&P 400)...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    def wiki_tickers(url, col):
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return pd.read_html(StringIO(r.text), header=0)[0][col].tolist()

    sp500 = wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"
    )
    try:
        sp400 = wiki_tickers(
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Ticker symbol"
        )
    except Exception:
        sp400 = []
    tickers = list(dict.fromkeys(
        t.replace(".", "-") for t in sp500 + sp400 if isinstance(t, str)
    ))
    print(f"  {len(tickers)} candidates (S&P 500 + S&P 400)")
    return tickers


def download_data(tickers):
    targets = tickers + (["SPY"] if "SPY" not in tickers else [])
    print(f"Downloading OHLCV ({DATA_START} to {SIM_END}) for {len(targets)} tickers...")
    raw = yf.download(
        targets,
        start=DATA_START.strftime("%Y-%m-%d"),
        end=(SIM_END + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=True,
        threads=False,
    )
    result = {}
    for ticker in targets:
        try:
            if len(targets) == 1:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].copy()
            df = df.dropna(how="all")
            if not df.empty:
                result[ticker] = df
        except Exception:
            pass
    print(f"  Data loaded for {len(result)} tickers\n")
    return result


def filter_universe_historically(tickers, price_data):
    valid = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None:
            continue
        warmup = df[df.index.date < SIM_START]
        if warmup.empty:
            continue
        if float(warmup["Close"].dropna().iloc[0]) < HIST_MIN_PRICE:
            continue
        if float(warmup["Volume"].dropna().mean()) < HIST_MIN_VOL:
            continue
        valid.append(ticker)
    print(f"  {len(valid)} stocks passed historical filters\n")
    return valid


# -------- SCORING (runs once, before SIM_START) --------

def score_all(tickers, price_data):
    print(f"Scoring {len(tickers)} stocks as of {SIM_START}...")
    rows = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None:
            continue
        hist = df[df.index.date < SIM_START].copy()
        if len(hist) < 60:
            continue

        close  = hist["Close"].astype(float)
        volume = hist["Volume"].astype(float)
        high   = hist["High"].astype(float)
        low    = hist["Low"].astype(float)

        sma20       = close.rolling(20).mean()
        sma50       = close.rolling(50).mean()
        ema12       = close.ewm(span=12, adjust=False).mean()
        ema26       = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        rsi         = compute_rsi(close)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1]) / float(close.iloc[-1])

        c  = float(close.iloc[-1])
        r  = float(rsi.iloc[-1])
        m1 = int(c > float(sma50.iloc[-1]))
        m2 = int(float(sma20.iloc[-1]) > float(sma50.iloc[-1]))
        m3 = int(40 <= r <= 68)
        m4 = int(float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1]))
        m5 = int(float(volume.rolling(5).mean().iloc[-1]) >
                 float(volume.rolling(20).mean().iloc[-1]))
        m6 = int(0.01 <= atr_pct <= 0.045)

        # m7: relative strength vs SPY over 63 trading days (~3 months)
        spy_hist = price_data.get("SPY")
        if spy_hist is not None:
            spy_pre = spy_hist[spy_hist.index.date < SIM_START]["Close"].astype(float)
            if len(spy_pre) >= 63 and len(close) >= 63:
                stock_rs = float(close.iloc[-1]) / float(close.iloc[-63]) - 1
                spy_rs   = float(spy_pre.iloc[-1]) / float(spy_pre.iloc[-63]) - 1
                m7 = int(stock_rs > spy_rs)
            else:
                m7 = 0
        else:
            m7 = 0

        rows.append({"Ticker": ticker, "Score": m1+m2+m3+m4+m5+m6+m7, "Price": c})

    df_scores = (pd.DataFrame(rows)
                 .sort_values(["Score", "Price"], ascending=[False, False])
                 .reset_index(drop=True))
    print(f"  {len(df_scores)} stocks scored  (top scorer: {df_scores.iloc[0]['Ticker']} score={df_scores.iloc[0]['Score']})\n")
    return df_scores


# -------- BACKTEST (called hundreds of times) --------

def run_backtest(selected, scores_df, price_data, stop_loss, take_profit, trailing_stop):
    alloc = INITIAL_CAPITAL / len(selected)
    cash  = 0.0

    class Pos:
        __slots__ = ["ticker","entry","shares","peak","exit_px","closed"]
        def __init__(self, ticker, entry, shares):
            self.ticker  = ticker
            self.entry   = entry
            self.shares  = shares
            self.peak    = entry
            self.exit_px = None
            self.closed  = False

    positions = []
    for ticker in selected:
        df = price_data.get(ticker)
        if df is None:
            continue
        ep, _ = entry_on(df, SIM_START)
        if ep:
            positions.append(Pos(ticker, ep, alloc / ep))

    initial_set = set(selected)
    candidate_queue = [
        t for t in scores_df["Ticker"].tolist()
        if t not in initial_set and t in price_data
    ]

    spy_df = price_data.get("SPY", list(price_data.values())[0])
    days   = sorted(d for d in spy_df.index.date if SIM_START <= d <= SIM_END)

    equity_curve = []
    for day in days:
        pending_reinvest = []
        for pos in positions:
            if pos.closed:
                continue
            px = price_on(price_data[pos.ticker], day)
            if px is None:
                continue
            if px > pos.peak:
                pos.peak = px
            chg   = (px - pos.entry) / pos.entry
            trail = (pos.peak - px) / pos.peak
            if chg <= -stop_loss or chg >= take_profit or trail >= trailing_stop:
                pos.exit_px = px
                pos.closed  = True
                freed = px * pos.shares
                cash += freed
                pending_reinvest.append(freed)

        if pending_reinvest:
            open_tickers = {p.ticker for p in positions if not p.closed}
            for invest_amount in pending_reinvest:
                while candidate_queue:
                    next_ticker = candidate_queue.pop(0)
                    if next_ticker in open_tickers:
                        continue
                    df = price_data.get(next_ticker)
                    if df is None:
                        continue
                    ep, _ = entry_on(df, day)
                    if ep is None:
                        continue
                    positions.append(Pos(next_ticker, ep, invest_amount / ep))
                    cash -= invest_amount
                    open_tickers.add(next_ticker)
                    break

        open_val = sum(
            (price_on(price_data[p.ticker], day) or p.entry) * p.shares
            for p in positions if not p.closed
        )
        equity_curve.append(cash + open_val)

    if not equity_curve:
        return 0.0, 0.0, 0.0

    final_val = equity_curve[-1]
    total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    eq        = pd.Series(equity_curve)
    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * (252**0.5) if daily_ret.std() > 0 else 0
    max_dd    = float(((eq - eq.cummax()) / eq.cummax()).min() * 100)

    return total_ret, sharpe, max_dd


# -------- SPY BENCHMARK --------

def spy_return(price_data):
    spy_df = price_data.get("SPY")
    if spy_df is None:
        return 0.0
    s0 = price_on(spy_df[spy_df.index.date <= SIM_START], SIM_START)
    s1 = price_on(spy_df[spy_df.index.date <= SIM_END], SIM_END)
    if s0 and s1:
        return (s1 - s0) / s0 * 100
    return 0.0


# -------- GRID SEARCH --------

def run_grid_search(scores_df, price_data, spy_ret):
    total_combos = (len(GRID["n_stocks"]) * len(GRID["stop_loss"])
                    * len(GRID["take_profit"]) * len(GRID["trailing_stop"]))
    print(f"Running grid search: {total_combos} parameter combinations...")
    print(f"SPY benchmark: {spy_ret:+.2f}%\n")

    results = []
    done    = 0

    for n, sl, tp, ts in product(
        GRID["n_stocks"], GRID["stop_loss"],
        GRID["take_profit"], GRID["trailing_stop"]
    ):
        # Trailing stop should not be tighter than stop loss (would dominate it)
        if ts > sl:
            done += 1
            continue

        selected = scores_df["Ticker"].head(n).tolist()
        if not selected:
            done += 1
            continue

        ret, sharpe, max_dd = run_backtest(selected, scores_df, price_data, sl, tp, ts)
        alpha = ret - spy_ret

        results.append({
            "n_stocks":      n,
            "stop_loss":     sl,
            "take_profit":   tp,
            "trailing_stop": ts,
            "return_%":      round(ret, 2),
            "alpha_%":       round(alpha, 2),
            "sharpe":        round(sharpe, 2),
            "max_dd_%":      round(max_dd, 2),
        })
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{total_combos} combos tested...", end="\r")

    print(f"  {done}/{total_combos} combos tested.       ")
    return pd.DataFrame(results)


# -------- MAIN --------

universe   = get_universe()
price_data = download_data(universe)
universe   = filter_universe_historically(universe, price_data)
scores_df  = score_all(universe, price_data)
spy_ret    = spy_return(price_data)

results_df = run_grid_search(scores_df, price_data, spy_ret)

# -------- REPORT --------
W = 80
print()
print("="*W)
print(f"  OPTIMIZATION RESULTS  |  SPY benchmark: {spy_ret:>+.2f}%  |  Period: {SIM_START} to {SIM_END}")
print("="*W)

beats_spy = results_df[results_df["alpha_%"] > 0].sort_values(
    ["alpha_%", "sharpe"], ascending=[False, False]
)

if beats_spy.empty:
    print("  No combination outperformed SPY in this period.")
    print("  Top 20 by return regardless:\n")
    top = results_df.sort_values("return_%", ascending=False).head(20)
else:
    print(f"  {len(beats_spy)} combinations beat SPY. Top 25 by alpha:\n")
    top = beats_spy.head(25)

print(f"  {'n':>3}  {'stop':>6}  {'tp':>6}  {'trail':>6}  "
      f"{'return':>8}  {'alpha':>8}  {'sharpe':>7}  {'maxDD':>7}")
print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*6}  "
      f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")
for _, r in top.iterrows():
    print(f"  {int(r.n_stocks):>3}  "
          f"{r.stop_loss*100:>5.0f}%  "
          f"{r.take_profit*100:>5.0f}%  "
          f"{r.trailing_stop*100:>5.0f}%  "
          f"{r['return_%']:>+7.2f}%  "
          f"{r['alpha_%']:>+7.2f}%  "
          f"{r.sharpe:>7.2f}  "
          f"{r['max_dd_%']:>+7.2f}%")
print("="*W)

best = top.iloc[0]
print(f"""
  BEST PARAMETERS TO UPDATE IN main.py:
    N_STOCKS          = {int(best.n_stocks)}
    STOP_LOSS_PCT     = {best.stop_loss}
    TAKE_PROFIT_PCT   = {best.take_profit}
    TRAILING_STOP_PCT = {best.trailing_stop}
    Expected return   : {best['return_%']:>+.2f}%
    Alpha vs SPY      : {best['alpha_%']:>+.2f}%
    Sharpe ratio      : {best.sharpe:.2f}
""")
