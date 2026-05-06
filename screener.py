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
    "RS_vs_SPY":   True,   # m9: stock 5-day return > SPY 5-day return
}

# Download window: 100 days covers SMA50 + RSI/MACD warmup + 63-day RS lookback
TODAY      = date.today()
DATA_START = TODAY - timedelta(days=100)

# SPDR sector ETFs used for the sector performance snapshot
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Financials":             "XLF",
    "Health Care":            "XLV",
    "Consumer Discretionary": "XLY",
    "Industrials":            "XLI",
    "Communication Svcs":     "XLC",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
}


# -------- HELPERS --------

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# -------- STEP 0: SECTOR ETF SNAPSHOT --------

def print_sector_etf_snapshot():
    """Download 11 SPDR sector ETFs and print 1D / 5D / 1M / 3M return table."""
    etf_tickers = list(SECTOR_ETFS.values())
    print("Fetching sector ETF performance...")
    raw = yf.download(
        etf_tickers,
        start=(TODAY - timedelta(days=90)).strftime("%Y-%m-%d"),
        end=(TODAY + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    rows = []
    for sector, etf in SECTOR_ETFS.items():
        try:
            df = raw[etf].copy() if len(etf_tickers) > 1 else raw.copy()
            close = df["Close"].dropna().astype(float)
            if len(close) < 2:
                continue
            c = float(close.iloc[-1])
            def ret(n): return (c / float(close.iloc[-n - 1]) - 1) * 100 if len(close) > n else 0.0
            rows.append({"Sector": sector, "ETF": etf,
                         "1D%": ret(1), "5D%": ret(5), "1M%": ret(21), "3M%": ret(63)})
        except Exception:
            continue

    df_etf = (pd.DataFrame(rows)
              .sort_values("1D%", ascending=False)
              .reset_index(drop=True))

    W = 80
    print()
    print("=" * W)
    print(f"  Sector ETF Performance  |  {TODAY}")
    print("=" * W)
    print(f"  {'Sector':<26} {'ETF':<5} {'1D':>7} {'5D':>7} {'1M':>7} {'3M':>8}  Momentum (3M)")
    print("-" * W)
    for _, r in df_etf.iterrows():
        v    = r["3M%"]
        sign = "▲" if v >= 0 else "▼"
        bar  = "|" * min(int(abs(v) / 2), 20)
        print(f"  {r['Sector']:<26} {r['ETF']:<5} "
              f"{r['1D%']:>+6.1f}% {r['5D%']:>+6.1f}% {r['1M%']:>+6.1f}% {r['3M%']:>+7.1f}%  {sign}{bar}")
    print("=" * W)
    print()


# -------- STEP 1: UNIVERSE --------

def _fetch_sec_tickers():
    """
    Pull all US exchange-listed equities from the SEC EDGAR company tickers endpoint.
    Returns a flat list of ticker symbols (NYSE + NASDAQ + ARCA etc.), excluding
    any entry whose name looks like an ETF/fund/trust/warrant/preferred share.
    Falls back to an empty list if unreachable.
    """
    EXCLUDE_KEYWORDS = (
        "etf", "fund", "trust", "warrant", "preferred", "notes", "bond",
        "debenture", "unit", "rights", "index", "acquisition",
    )
    headers = {"User-Agent": "InvestorOz/1.0 contact@example.com"}
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers_exchange.json",
            headers=headers, timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        # Format: {"fields": [...], "data": [[cik, name, ticker, exchange], ...]}
        fields = data["fields"]
        ticker_idx   = fields.index("ticker")
        name_idx     = fields.index("name")
        exchange_idx = fields.index("exchange")

        VALID_EXCHANGES = {"nyse", "nasdaq", "nysearca", "bats", "cboe"}
        tickers = []
        for row in data["data"]:
            exchange = str(row[exchange_idx]).lower()
            if exchange not in VALID_EXCHANGES:
                continue
            name   = str(row[name_idx]).lower()
            ticker = str(row[ticker_idx]).strip().replace(".", "-")
            if not ticker or not ticker.replace("-", "").isalpha():
                continue
            if any(kw in name for kw in EXCLUDE_KEYWORDS):
                continue
            tickers.append(ticker)
        return tickers
    except Exception:
        return []


def get_universe():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def wiki_tickers(url, col):
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return pd.read_html(StringIO(r.text), header=0)[0][col].tolist()

    # --- S&P Composite 1500 (always attempted) ---
    print("Fetching S&P 500 + S&P 400 + S&P 600 from Wikipedia...")
    sp500 = wiki_tickers(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"
    )
    try:
        sp400 = wiki_tickers(
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Ticker symbol"
        )
    except Exception:
        sp400 = []
    try:
        sp600 = wiki_tickers(
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Ticker symbol"
        )
    except Exception:
        sp600 = []

    # --- Full NYSE + NASDAQ listings from SEC EDGAR ---
    print("Fetching full NYSE + NASDAQ listings from SEC EDGAR...")
    exchange_tickers = _fetch_sec_tickers()
    if exchange_tickers:
        print(f"  {len(exchange_tickers)} raw tickers from SEC EDGAR")
    else:
        print("  SEC EDGAR unavailable - falling back to S&P Composite 1500 only")

    # Combine all sources; preserve order (S&P first = priority for sector map)
    all_raw = sp500 + sp400 + sp600 + exchange_tickers
    tickers = list(dict.fromkeys(
        t.replace(".", "-") for t in all_raw
        if isinstance(t, str) and t.isalpha() or (isinstance(t, str) and "-" in t)
    ))
    print(f"  {len(tickers)} unique candidates after dedup")
    return tickers



def get_sector_map():
    """Return {ticker: GICS sector} for S&P 500 stocks from Wikipedia."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=15
        )
        r.raise_for_status()
        table = pd.read_html(StringIO(r.text), header=0)[0]
        return {
            str(row["Symbol"]).replace(".", "-"): str(row["GICS Sector"])
            for _, row in table.iterrows()
        }
    except Exception:
        return {}


# -------- STEP 2: DOWNLOAD PRICE DATA --------

def download_data(tickers):
    targets = list(dict.fromkeys(tickers + ["SPY"]))  # ensure SPY always present
    print(f"\nDownloading OHLCV ({DATA_START} to {TODAY}) for {len(targets)} tickers...")
    raw = yf.download(
        targets,
        start=DATA_START.strftime("%Y-%m-%d"),
        end=(TODAY + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=True,
        threads=True,
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
    print(f"  Data loaded for {len(result)} tickers")
    return result


# -------- STEP 3: FILTER & SCORE --------

def score_universe(tickers, price_data, sector_map=None):
    rows = []

    # Pre-compute SPY 5-day return for RS_vs_SPY metric
    spy_df    = price_data.get("SPY")
    spy_close = spy_df["Close"].astype(float) if spy_df is not None else None
    spy_ret5  = (
        float(spy_close.iloc[-1]) / float(spy_close.iloc[-5]) - 1
        if spy_close is not None and len(spy_close) >= 5 else 0.0
    )

    for ticker in tickers:
        if ticker == "SPY":
            continue  # never score SPY itself as a candidate
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
        stock_ret5 = (
            float(close.iloc[-1]) / float(close.iloc[-5]) - 1
            if len(close) >= 5 else 0.0
        )
        m9 = int(stock_ret5 > spy_ret5)

        # Apply toggles
        m1 = m1 if METRICS_ENABLED["AboveSMA50"]  else 0
        m2 = m2 if METRICS_ENABLED["MA_Aligned"]   else 0
        m3 = m3 if METRICS_ENABLED["RSI_Healthy"]  else 0
        m4 = m4 if METRICS_ENABLED["MACD_Bull"]    else 0
        m5 = m5 if METRICS_ENABLED["VolExpand"]    else 0
        m6 = m6 if METRICS_ENABLED["ATR_OK"]       else 0
        m7 = m7 if METRICS_ENABLED["NearEMA10"]    else 0
        m8 = m8 if METRICS_ENABLED["EMACoil"]      else 0
        m9 = m9 if METRICS_ENABLED["RS_vs_SPY"]    else 0

        # RS_3m: 10-day return — used for ranking, not pass/fail
        rs_3m = (float(close.iloc[-1]) / float(close.iloc[-10]) - 1) * 100 \
            if len(close) >= 10 else 0.0

        rows.append({
            "Ticker":      ticker,
            "Sector":      sector_map.get(ticker, "Other") if sector_map else "Unknown",
            "Score":       m1+m2+m3+m4+m5+m6+m7+m8+m9,
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
            "RS_vs_SPY":   bool(m9),
            "RS_3m":       round(rs_3m, 1),
        })

    return (pd.DataFrame(rows)
            .sort_values(["Score", "RS_3m"], ascending=[False, False])
            .reset_index(drop=True))


# -------- STEP 4a: SECTOR HEAT MAP --------

def print_sector_heatmap(df):
    """Group the scored universe by GICS sector and print avg momentum + setup quality.
    Returns a list of sector names sorted by AvgRS3m (best first)."""
    if "Sector" not in df.columns:
        return []
    max_score = len([v for v in METRICS_ENABLED.values() if v])
    grp = (
        df.groupby("Sector")
        .agg(
            Stocks     =("Ticker",  "count"),
            AvgScore   =("Score",   "mean"),
            AvgRS3m    =("RS_3m",   "mean"),
            HighSetups =("Score",   lambda x: (x >= max(max_score - 2, 1)).sum()),
        )
        .reset_index()
        .sort_values("AvgRS3m", ascending=False)
    )

    W = 80
    print()
    print("=" * W)
    print(f"  Sector Heat Map (Universe)  |  {TODAY}")
    print(f"  High-setup = Score ≥ {max(max_score - 2, 1)}/{max_score}")
    print("=" * W)
    print(f"  {'Sector':<26} {'#':>4}  {'AvgScr':>8}  {'AvgRS3m':>8}  {'SetupCnt':>8}  Momentum (3M avg)")
    print("-" * W)
    for _, r in grp.iterrows():
        v    = r["AvgRS3m"]
        sign = "▲" if v >= 0 else "▼"
        bar  = "|" * min(int(abs(v) / 2), 20)
        print(f"  {r['Sector']:<26} {int(r['Stocks']):>4}  "
              f"{r['AvgScore']:>5.1f}/{max_score:<2}  "
              f"{r['AvgRS3m']:>+7.1f}%  "
              f"{int(r['HighSetups']):>8}  {sign}{bar}")
    print("=" * W)
    print()
    return grp["Sector"].tolist()


# -------- STEP 4b: PRINT RESULTS --------

def print_results(df, sectors=None):
    enabled_metrics = [k for k, v in METRICS_ENABLED.items() if v]
    max_score = len(enabled_metrics)

    W = 90
    sector_label = f"  Sectors: {' | '.join(sectors)}" if sectors else ""
    print()
    print("=" * W)
    print(f"  InvestorOz Live Screener  |  {TODAY}  |  Top {TOP_N} candidates")
    if sector_label:
        print(sector_label)
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
    header = (f"{'#':>3}  {'Ticker':<8} {'Score':>6}  {'Price':>8}  {'RSI':>5}  RS10d  "
              f"{'SMA50':>6} {'MAlign':>6} {'RSI':>5} {'MACD':>5} "
              f"{'Vol':>5} {'ATR':>5} {'NrE10':>6} {'Coil':>5} {'vSPY':>5}")
    print(header)
    print("-" * W)
    for i, row in top.iterrows():
        def b(v): return " YES" if v else "  no"
        line = (f"{i+1:>3}  {row['Ticker']:<8} {int(row['Score']):>3}/{max_score:<2}  "
                f"${row['Price']:>7.2f}  {row['RSI']:>5.1f}  "
                f"{row['RS_3m']:>+5.1f}%  "
                f"{b(row['AboveSMA50'])} {b(row['MA_Aligned'])} {b(row['RSI_Healthy'])} "
                f"{b(row['MACD_Bull'])} {b(row['VolExpand'])} {b(row['ATR_OK'])} "
                f"{b(row['NearEMA10'])} {b(row['EMACoil'])} {b(row['RS_vs_SPY'])}")
        print(line)

    print("=" * W)
    total_pass = (df["Score"] == max_score).sum()
    print(f"  {len(df)} stocks scored  |  {total_pass} passed all {max_score} active metrics")
    print("=" * W)
    print()


# -------- MAIN --------

if __name__ == "__main__":
    print_sector_etf_snapshot()
    tickers    = get_universe()
    sector_map = get_sector_map()
    price_data = download_data(tickers)
    print("\nScoring stocks on live data...")
    scores_df   = score_universe(tickers, price_data, sector_map)
    all_sectors = print_sector_heatmap(scores_df)
    top2        = all_sectors[:2]
    filtered_df = scores_df[scores_df["Sector"].isin(top2)].reset_index(drop=True)
    print_results(filtered_df, sectors=top2)
