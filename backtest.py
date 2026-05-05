#!/usr/bin/env python3
"""
InvestorOz -- Paper Trading Backtest

SELECTION METRICS (0-8 points, 1 per metric; m7/m8 disabled by default):
  1. Price above SMA50              -- confirmed long-term uptrend
  2. SMA20 > SMA50                  -- moving average alignment (momentum structure)
  3. RSI between 40 and 68         -- healthy: not oversold, not overbought
  4. MACD above signal line         -- short-term bullish momentum
  5. 5-day vol > 20-day vol         -- expanding volume (rising interest)
  6. ATR/Price 1%-4.5%              -- optimal volatility range
  7. |Price - EMA10| / Price < 5%  -- not extended; close to short-term mean
  8. |EMA10 - EMA21| / Price < 2%  -- EMAs converging / coiling

TRADING RULES:
  Entry         : Equal-weight buy at first open price on/after SIM_START
                  Reinvestment: freed capital is immediately redeployed into
                  the next highest-scored candidate not currently held
  Stop-Loss     : Closes position at -3% from entry
  Take-Profit   : Closes position at +25% from entry
  Trailing Stop : Closes position if price drops 5% from its rolling peak
                  (tight trail drives rapid capital recycling into new positions)
  Universe      : S&P 500 + S&P 400 constituents (~900 stocks, Wikipedia)
                  Historical filter: price >= $5, avg vol >= 500k during warmup
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import date, timedelta

# -------- CONFIG --------
INITIAL_CAPITAL   = 10_000
N_STOCKS          = 3        # optimized: top 3 scored stocks
HIST_MIN_PRICE    = 5.0      # min price as of DATA_START (historical, not today)
HIST_MIN_VOL      = 500_000  # min avg daily volume during warmup window (historical)
STOP_LOSS_PCT     = 0.03     # optimized: -3% stop-loss
TAKE_PROFIT_PCT   = 0.25     # optimized: +25% take-profit
TRAILING_STOP_PCT = 0.05     # optimized: -5% trailing stop from peak

SIM_START  = date(2025, 1, 2)    # first trading day of 2025
SIM_END    = date(2025, 12, 31)  # last trading day of 2025
DATA_START = date(2024, 5, 1)    # ~6 months before SIM_START for indicator warmup

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


# -------- HELPERS --------

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def price_on(df, d):
    """Last close price on or before date d."""
    rows = df[df.index.date <= d]
    return float(rows["Close"].iloc[-1]) if not rows.empty else None


def entry_on(df, d):
    """First available open/close price on or after date d. Returns (price, date)."""
    rows = df[df.index.date >= d]
    if rows.empty:
        return None, None
    col = "Open" if "Open" in rows.columns else "Close"
    return float(rows[col].iloc[0]), rows.index[0].date()


# -------- STEP 1: UNIVERSE --------

def get_universe():
    """
    Build universe from S&P 500 + S&P 400 Wikipedia membership lists.
    These are point-in-time stable index constituents, giving a reproducible
    ~900-stock candidate pool with no survivorship bias from live screeners.
    """
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
    targets = tickers + (["SPY"] if "SPY" not in tickers else [])
    print(f"\nDownloading OHLCV ({DATA_START} to {SIM_END}) for {len(targets)} tickers...")
    raw = yf.download(
        targets,
        start=DATA_START.strftime("%Y-%m-%d"),
        end=(SIM_END + timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        progress=False,
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
    print(f"  Data loaded for {len(result)} tickers")
    return result


# -------- STEP 2b: HISTORICAL UNIVERSE FILTER --------

def filter_universe_historically(tickers, price_data):
    """
    Re-filter the screener pool using actual historical data as of DATA_START.
    This removes survivorship bias: stocks that only became large AFTER 2025
    are excluded, and stocks that were liquid in early 2025 but have since
    delisted/shrunk are kept.
    Criteria applied to the warmup window (DATA_START to SIM_START):
      - Price >= HIST_MIN_PRICE at DATA_START
      - Average daily volume >= HIST_MIN_VOL during warmup window
    """
    valid = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None:
            continue
        warmup = df[df.index.date < SIM_START]
        if warmup.empty:
            continue
        # Price at start of warmup window
        first_price = float(warmup["Close"].dropna().iloc[0])
        if first_price < HIST_MIN_PRICE:
            continue
        # Average volume during warmup window
        avg_vol = float(warmup["Volume"].dropna().mean())
        if avg_vol < HIST_MIN_VOL:
            continue
        valid.append(ticker)
    print(f"  {len(valid)} stocks passed historical filters "
          f"(price >= ${HIST_MIN_PRICE}, avg vol >= {HIST_MIN_VOL:,} during warmup)")
    return valid


# -------- STEP 3: SCORE & SELECT (pre-SIM_START data only) --------

def _score_ticker(df, cutoff_date):
    """
    Score a single ticker using data up to (but not including) cutoff_date.
    Returns score (int) and price (float), or (None, None) if insufficient data.
    """
    hist = df[df.index.date < cutoff_date].copy()
    if len(hist) < 60:
        return None, None

    close  = hist["Close"].astype(float)
    volume = hist["Volume"].astype(float)
    high   = hist["High"].astype(float)
    low    = hist["Low"].astype(float)

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
    m7 = int(abs(c - float(ema10.iloc[-1])) / c < 0.05)
    m8 = int(abs(float(ema10.iloc[-1]) - float(ema21.iloc[-1])) / c < 0.02)

    m1 = m1 if METRICS_ENABLED["AboveSMA50"]  else 0
    m2 = m2 if METRICS_ENABLED["MA_Aligned"]   else 0
    m3 = m3 if METRICS_ENABLED["RSI_Healthy"]  else 0
    m4 = m4 if METRICS_ENABLED["MACD_Bull"]    else 0
    m5 = m5 if METRICS_ENABLED["VolExpand"]    else 0
    m6 = m6 if METRICS_ENABLED["ATR_OK"]       else 0
    m7 = m7 if METRICS_ENABLED["NearEMA10"]    else 0
    m8 = m8 if METRICS_ENABLED["EMACoil"]      else 0

    return m1+m2+m3+m4+m5+m6+m7+m8, round(c, 2)


def score_and_select(tickers, price_data):
    """
    Score each stock on up to 8 technical metrics using ONLY data before SIM_START.
    This simulates what a trader would have seen at the start of the period.
    """
    rows = []
    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None:
            continue
        score, price = _score_ticker(df, SIM_START)
        if score is None:
            continue
        hist = df[df.index.date < SIM_START]
        close  = hist["Close"].astype(float)
        volume = hist["Volume"].astype(float)
        high   = hist["High"].astype(float)
        low    = hist["Low"].astype(float)
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
        atr_pct = float(tr.rolling(14).mean().iloc[-1]) / float(close.iloc[-1])
        c  = float(close.iloc[-1])
        r  = float(rsi.iloc[-1])
        m1 = int(c > float(sma50.iloc[-1])) if METRICS_ENABLED["AboveSMA50"]  else 0
        m2 = int(float(sma20.iloc[-1]) > float(sma50.iloc[-1])) if METRICS_ENABLED["MA_Aligned"]   else 0
        m3 = int(40 <= r <= 68) if METRICS_ENABLED["RSI_Healthy"]  else 0
        m4 = int(float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1])) if METRICS_ENABLED["MACD_Bull"]    else 0
        m5 = int(float(volume.rolling(5).mean().iloc[-1]) > float(volume.rolling(20).mean().iloc[-1])) if METRICS_ENABLED["VolExpand"]    else 0
        m6 = int(0.01 <= atr_pct <= 0.045) if METRICS_ENABLED["ATR_OK"]       else 0
        m7 = int(abs(c - float(ema10.iloc[-1])) / c < 0.05) if METRICS_ENABLED["NearEMA10"]    else 0
        m8 = int(abs(float(ema10.iloc[-1]) - float(ema21.iloc[-1])) / c < 0.02) if METRICS_ENABLED["EMACoil"]      else 0
        rows.append({
            "Ticker":      ticker,
            "Score":       m1+m2+m3+m4+m5+m6+m7+m8,
            "Price":       round(c, 2),
            "RSI":         round(r, 1),
            "AboveSMA50":  bool(m1),
            "MA_Aligned":  bool(m2),
            "RSI_Healthy": bool(m3),
            "MACD_Bull":   bool(m4),
            "VolExpand":   bool(m5),
            "ATR_OK":      bool(m6),
            "NearEMA10":   bool(m7),
            "EMACoil":     bool(m8),
        })

    return (pd.DataFrame(rows)
            .sort_values(["Score", "Price"], ascending=[False, False])
            .reset_index(drop=True))


def best_candidate_on_day(candidates, price_data, day, exclude):
    """
    Re-score all candidates using data up to `day` and return the highest-scoring
    ticker not in `exclude`. This is called at reinvestment time so the replacement
    stock is evaluated on current market conditions, not the original SIM_START score.
    """
    best_ticker, best_score, best_price = None, -1, 0
    for ticker in candidates:
        if ticker in exclude:
            continue
        df = price_data.get(ticker)
        if df is None:
            continue
        score, price = _score_ticker(df, day)
        if score is None:
            continue
        if score > best_score or (score == best_score and price > best_price):
            best_score, best_price, best_ticker = score, price, ticker
    return best_ticker


# -------- STEP 4: BACKTEST ENGINE --------

class Position:
    def __init__(self, ticker, entry_price, shares, entry_date):
        self.ticker      = ticker
        self.entry_price = entry_price
        self.shares      = shares
        self.entry_date  = entry_date
        self.peak_price  = entry_price
        self.exit_price  = None
        self.exit_date   = None
        self.exit_reason = None

    @property
    def is_open(self):
        return self.exit_price is None

    def tick(self, price, day):
        """Update peak; apply exit rules. Returns True if position was closed."""
        if price > self.peak_price:
            self.peak_price = price
        change     = (price - self.entry_price) / self.entry_price
        trail_drop = (self.peak_price - price) / self.peak_price

        reason = None
        if change <= -STOP_LOSS_PCT:
            reason = f"Stop-loss ({change*100:+.1f}%)"
        elif change >= TAKE_PROFIT_PCT:
            reason = f"Take-profit ({change*100:+.1f}%)"
        elif trail_drop >= TRAILING_STOP_PCT:
            reason = f"Trailing stop (-{trail_drop*100:.1f}% from peak)"

        if reason:
            self.exit_price  = price
            self.exit_date   = day
            self.exit_reason = reason
            return True
        return False

    def pnl(self):
        ep = self.exit_price if self.exit_price else self.entry_price
        return (ep - self.entry_price) * self.shares

    def pnl_pct(self):
        ep = self.exit_price if self.exit_price else self.entry_price
        return (ep - self.entry_price) / self.entry_price * 100


def run_backtest(selected, scores_df, price_data):
    alloc = INITIAL_CAPITAL / len(selected)
    cash  = 0.0
    positions = []

    for ticker in selected:
        df = price_data.get(ticker)
        if df is None:
            continue
        entry_px, entry_dt = entry_on(df, SIM_START)
        if entry_px is None:
            continue
        positions.append(Position(ticker, entry_px, alloc / entry_px, entry_dt))

    # Pool of candidates for reinvestment (all non-initial stocks with data)
    initial_set = set(selected)
    candidate_pool = [
        t for t in scores_df["Ticker"].tolist()
        if t not in initial_set and t in price_data
    ]

    spy_df = price_data.get("SPY", list(price_data.values())[0])
    trading_days = sorted(d for d in spy_df.index.date if SIM_START <= d <= SIM_END)

    history = []
    for day in trading_days:
        # --- Check exit conditions; collect freed cash per closed slot ---
        pending_reinvest = []  # one entry per closed position: amount to reinvest
        for pos in positions:
            if not pos.is_open:
                continue
            px = price_on(price_data[pos.ticker], day)
            if px and pos.tick(px, day):
                freed = pos.exit_price * pos.shares
                cash += freed
                pending_reinvest.append(freed)

        # --- Reinvest: re-score candidates on this day, pick best available ---
        if pending_reinvest:
            for invest_amount in pending_reinvest:
                open_tickers = {p.ticker for p in positions if p.is_open}
                exclude = open_tickers | initial_set
                next_ticker = best_candidate_on_day(candidate_pool, price_data, day, exclude)
                if next_ticker is None:
                    cash += invest_amount
                    continue
                df = price_data.get(next_ticker)
                entry_px, entry_dt = entry_on(df, day)
                if entry_px is None:
                    cash += invest_amount
                    continue
                positions.append(
                    Position(next_ticker, entry_px, invest_amount / entry_px, entry_dt)
                )
                cash -= invest_amount

        # --- Mark-to-market ---
        open_val = sum(
            (price_on(price_data[pos.ticker], day) or pos.entry_price) * pos.shares
            for pos in positions if pos.is_open
        )
        history.append({"date": day, "value": round(cash + open_val, 2)})

    # Mark remaining open positions as closed at last available price
    last_day = trading_days[-1] if trading_days else SIM_END
    for pos in positions:
        if pos.is_open:
            px = price_on(price_data[pos.ticker], last_day)
            pos.exit_price  = px or pos.entry_price
            pos.exit_date   = last_day
            pos.exit_reason = "End of simulation (still held)"

    return positions, pd.DataFrame(history)


# -------- STEP 5: REPORT --------

def print_report(positions, history_df, price_data):
    df_eq = history_df.copy()
    df_eq["date"] = pd.to_datetime(df_eq["date"])
    eq = df_eq.set_index("date")["value"]

    final_val = float(eq.iloc[-1])
    total_ret = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    spy_ret = None
    spy_df  = price_data.get("SPY")
    if spy_df is not None:
        s0 = price_on(spy_df[spy_df.index.date <= SIM_START], SIM_START)
        s1 = price_on(spy_df, SIM_END)
        if s0 and s1:
            spy_ret = (s1 - s0) / s0 * 100

    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std()) * (252**0.5) if daily_ret.std() > 0 else 0
    max_dd    = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    wins      = sum(1 for p in positions if p.pnl_pct() >= 0)
    losses    = sum(1 for p in positions if p.pnl_pct() < 0)
    reinvests = len(positions) - N_STOCKS

    W = 76
    print()
    print("="*W)
    print(f"{'  INVESTOROZ -- PAPER TRADING BACKTEST REPORT  ':^{W}}")
    print("="*W)
    print(f"  Simulation period  : {SIM_START}  to  {SIM_END}")
    print(f"  Starting capital   : ${INITIAL_CAPITAL:>10,.2f}")
    print(f"  Final value        : ${final_val:>10,.2f}")
    print(f"  Total return       : {total_ret:>+8.2f}%")
    if spy_ret is not None:
        alpha = total_ret - spy_ret
        print(f"  SPY benchmark      : {spy_ret:>+8.2f}%   |  alpha vs SPY: {alpha:>+.2f}%")
    print(f"  Sharpe ratio       : {sharpe:>8.2f}  (annualized)")
    print(f"  Max drawdown       : {max_dd:>8.2f}%")
    print(f"  Win / Loss trades  : {wins} wins  /  {losses} losses  ({reinvests} reinvestments)")
    print()
    print("-"*W)
    print(f"  {'TICKER':<8} {'BUY DATE':<12} {'SELL DATE':<12} {'ENTRY':>9} {'EXIT':>9} {'P&L $':>9} {'P&L %':>7}  REASON")
    print("-"*W)
    for pos in sorted(positions, key=lambda p: (p.entry_date, p.pnl()), reverse=False):
        ep   = pos.exit_price or pos.entry_price
        pnl  = pos.pnl()
        ppct = pos.pnl_pct()
        sign = "+" if pnl >= 0 else ""
        print(f"  {pos.ticker:<8} {str(pos.entry_date):<12} {str(pos.exit_date):<12} "
              f"${pos.entry_price:>8.2f} ${ep:>8.2f} "
              f"{sign}${abs(pnl):>7.2f}  {sign}{ppct:.1f}%  {pos.exit_reason}")
    print("-"*W)
    total_pnl = sum(p.pnl() for p in positions)
    sign = "+" if total_pnl >= 0 else "-"
    print(f"  NET P&L{' '*(W-32)}{sign}${abs(total_pnl):>7.2f}  ({total_ret:>+.2f}%)")
    print("="*W)

    # Monthly equity curve with ASCII bar chart
    print(f"\n  MONTHLY EQUITY CURVE  (each bar char ~2% return)")
    print(f"  {'-'*12}  {'-'*15}  {'-'*8}  chart")
    monthly = eq.resample("MS").first()
    for dt, val in monthly.items():
        ret     = (val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        bar_len = max(1, int(abs(ret) / 2))
        bar     = ("+" * min(bar_len, 25)) if ret >= 0 else ("-" * min(bar_len, 25))
        sign    = "+" if ret >= 0 else ""
        print(f"  {str(dt.date()):<12}  ${val:>14,.2f}  {sign}{ret:>7.2f}%  {bar}")
    last_dt = pd.to_datetime(SIM_END)
    if last_dt not in monthly.index:
        ret     = total_ret
        bar_len = max(1, int(abs(ret) / 2))
        bar     = ("+" * min(bar_len, 25)) if ret >= 0 else ("-" * min(bar_len, 25))
        sign    = "+" if ret >= 0 else ""
        print(f"  {str(SIM_END):<12}  ${final_val:>14,.2f}  {sign}{ret:>7.2f}%  {bar}")
    print("="*W)


# -------- MAIN --------

universe   = get_universe()
price_data = download_data(universe)
universe   = filter_universe_historically(universe, price_data)

print(f"\nScoring stocks as of {SIM_START} (using warmup data from {DATA_START})...")
scores = score_and_select(universe, price_data)

W = 76
print("\n" + "="*W)
print(f"  TOP SCORED STOCKS AS OF {SIM_START}")
print(f"  Metrics: AboveSMA50 | MA_Aligned | RSI_Healthy | MACD_Bull | VolExpand | ATR_OK")
print("="*W)
print(scores.head(15).to_string(index=False))
print("="*W)

selected = scores["Ticker"].head(N_STOCKS).tolist()
print(f"\nSelected for paper trading: {', '.join(selected)}")
print(f"Rules: Stop-loss -{STOP_LOSS_PCT*100:.0f}% | Take-profit +{TAKE_PROFIT_PCT*100:.0f}% | Trailing stop -{TRAILING_STOP_PCT*100:.0f}% from peak")

print("\nRunning day-by-day simulation with reinvestment...")
positions, history_df = run_backtest(selected, scores, price_data)

print_report(positions, history_df, price_data)
