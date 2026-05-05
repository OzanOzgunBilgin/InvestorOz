#!/usr/bin/env python3
"""
InvestorOz -- Live Stock Screener

Scores every S&P 500 + S&P 400 stock against 6 technical metrics
using the most recent available market data and prints the top
candidates to buy today.

SELECTION METRICS (0-8 points, 1 per metric):
  1. Price above SMA50        -- confirmed long-term uptrend
  2. SMA20 > SMA50            -- moving average alignment (momentum structure)
  3. RSI between 40 and 68   -- healthy: not oversold, not overbought
  4. MACD above signal line   -- short-term bullish momentum
  5. 5-day vol > 20-day vol   -- expanding volume (rising interest)
  6. ATR/Price 1%-4.5%        -- optimal volatility range
  7. Price within 5% of EMA10 -- not extended; close to short-term mean
  8. EMA10 and EMA21 within 2% of price -- EMAs converging (coiling)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import date, timedelta

# -------- CONFIG --------
TOP_N          = 10          # number of top candidates to display
MIN_PRICE      = 5.0         # minimum current price
MIN_AVG_VOL    = 500_000     # minimum average daily volume (20-day)

# Toggle individual selection metrics on/off
METRICS_ENABLED = {
    "AboveSMA50":  True,   # m1: price > 50-day SMA
    "MA_Aligned":  True,   # m2: 20-day SMA > 50-day SMA
    "RSI_Healthy": True,   # m3: RSI between 40 and 68
    "MACD_Bull":   True,   # m4: MACD line > signal line
    "VolExpand":   True,   # m5: 5-day avg volume > 20-day avg volume
    "ATR_OK":      True,   # m6: ATR/Price between 1% and 4.5%
    "NearEMA10":   True,   # m7: |price - EMA10| / price < 5%
    "EMACoil":     True,   # m8: |EMA10 - EMA21| / price < 2%
}

# Download window: 100 days covers SMA50 + RSI/MACD warmup + 63-day RS lookback
TODAY      = date.today()
DATA_START = TODAY - timedelta(days=100)


# -------- HELPERS --------

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# -------- STEP 1: UNIVERSE --------

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


# -------- STEP 2: DOWNLOAD PRICE DATA --------

def download_data(tickers):
    print(f"\nDownloading OHLCV ({DATA_START} to {TODAY}) for {len(tickers)} tickers...")
    raw = yf.download(
        tickers,
        start=DATA_START.strftime("%Y-%m-%d"),
        end=(TODAY + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    result = {}
    for ticker in tickers:
        try:
            if len(tickers) == 1:
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
    print(f"  Data loaded for {len(result)} tickers")
    return result


# -------- STEP 3: FILTER & SCORE --------

def score_universe(tickers, price_data):
    rows = []

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or len(df) < 60:
            continue

        close  = df["Close"].astype(float)
        volume = df["Volume"].astype(float)
        high   = df["High"].astype(float)
        low    = df["Low"].astype(float)

        # Basic filters on latest data
        current_price  = float(close.iloc[-1])
        avg_vol_20     = float(volume.rolling(20).mean().iloc[-1])
        if current_price < MIN_PRICE or avg_vol_20 < MIN_AVG_VOL:
            continue

        # Indicators
        sma20       = close.rolling(20).mean()
        sma50       = close.rolling(50).mean()
        ema10       = close.ewm(span=10, adjust=False).mean()
        ema12       = close.ewm(span=12, adjust=False).mean()
        ema21       = close.ewm(span=21, adjust=False).mean()
        ema26       = close.ewm(span=26, adjust=False).mean()
        macd_line   = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        rsi         = compute_rsi(close)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1]) / current_price

        r  = float(rsi.iloc[-1])
        m1 = int(current_price > float(sma50.iloc[-1]))
        m2 = int(float(sma20.iloc[-1]) > float(sma50.iloc[-1]))
        m3 = int(40 <= r <= 68)
        m4 = int(float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1]))
        m5 = int(float(volume.rolling(5).mean().iloc[-1]) > avg_vol_20)
        m6 = int(0.01 <= atr_pct <= 0.045)
        m7 = int(abs(current_price - float(ema10.iloc[-1])) / current_price < 0.05)
        m8 = int(abs(float(ema10.iloc[-1]) - float(ema21.iloc[-1])) / current_price < 0.02)

        # Apply toggles
        m1 = m1 if METRICS_ENABLED["AboveSMA50"]  else 0
        m2 = m2 if METRICS_ENABLED["MA_Aligned"]   else 0
        m3 = m3 if METRICS_ENABLED["RSI_Healthy"]  else 0
        m4 = m4 if METRICS_ENABLED["MACD_Bull"]    else 0
        m5 = m5 if METRICS_ENABLED["VolExpand"]    else 0
        m6 = m6 if METRICS_ENABLED["ATR_OK"]       else 0
        m7 = m7 if METRICS_ENABLED["NearEMA10"]    else 0
        m8 = m8 if METRICS_ENABLED["EMACoil"]      else 0

        # RS_3m: 3-month return (63 trading days) — used for ranking, not pass/fail
        rs_3m = (float(close.iloc[-1]) / float(close.iloc[-63]) - 1) * 100 \
            if len(close) >= 63 else 0.0

        rows.append({
            "Ticker":      ticker,
            "Score":       m1+m2+m3+m4+m5+m6+m7+m8,
            "Price":       round(current_price, 2),
            "RSI":         round(r, 1),
            "AboveSMA50":  bool(m1),
            "MA_Aligned":  bool(m2),
            "RSI_Healthy": bool(m3),
            "MACD_Bull":   bool(m4),
            "VolExpand":   bool(m5),
            "ATR_OK":      bool(m6),
            "NearEMA10":   bool(m7),
            "EMACoil":     bool(m8),
            "RS_3m":       round(rs_3m, 1),
        })

    return (pd.DataFrame(rows)
            .sort_values(["Score", "RS_3m"], ascending=[False, False])
            .reset_index(drop=True))


# -------- STEP 4: PRINT RESULTS --------

def print_results(df):
    enabled_metrics = [k for k, v in METRICS_ENABLED.items() if v]
    max_score = len(enabled_metrics)

    W = 90
    print()
    print("=" * W)
    print(f"  InvestorOz Live Screener  |  {TODAY}  |  Top {TOP_N} candidates")
    print("=" * W)

    # Active / inactive metrics
    on  = "  ON : " + ", ".join(k for k, v in METRICS_ENABLED.items() if v)
    off_list = [k for k, v in METRICS_ENABLED.items() if not v]
    off = ("  OFF: " + ", ".join(off_list)) if off_list else ""
    print(on)
    if off:
        print(off)
    print("-" * W)

    top = df.head(TOP_N)
    header = (f"{'#':>3}  {'Ticker':<8} {'Score':>6}  {'Price':>8}  {'RSI':>5}  RS_3m  "
              f"{'SMA50':>6} {'MAlign':>6} {'RSI':>5} {'MACD':>5} "
              f"{'Vol':>5} {'ATR':>5} {'NrE10':>6} {'Coil':>5}")
    print(header)
    print("-" * W)
    for i, row in top.iterrows():
        def b(v): return " YES" if v else "  no"
        line = (f"{i+1:>3}  {row['Ticker']:<8} {int(row['Score']):>3}/{max_score:<2}  "
                f"${row['Price']:>7.2f}  {row['RSI']:>5.1f}  "
                f"{row['RS_3m']:>+5.1f}%  "
                f"{b(row['AboveSMA50'])} {b(row['MA_Aligned'])} {b(row['RSI_Healthy'])} "
                f"{b(row['MACD_Bull'])} {b(row['VolExpand'])} {b(row['ATR_OK'])} "
                f"{b(row['NearEMA10'])} {b(row['EMACoil'])}")
        print(line)

    print("=" * W)
    total_pass = (df["Score"] == max_score).sum()
    print(f"  {len(df)} stocks scored  |  {total_pass} passed all {max_score} active metrics")
    print("=" * W)
    print()


# -------- MAIN --------

if __name__ == "__main__":
    tickers    = get_universe()
    price_data = download_data(tickers)
    print("\nScoring stocks on live data...")
    scores_df  = score_universe(tickers, price_data)
    print_results(scores_df)
