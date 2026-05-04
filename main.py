import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MIN_AVG_VOLUME  = 500_000    # liquidity filter
MIN_PRICE       = 1.0        # filter out penny stocks
MIN_MARKET_CAP  = 50_000_000 # $50M minimum market cap
LOOKBACK        = 120        # price history window (trading days)
BATCH_SIZE      = 100        # tickers per yfinance bulk download
SCREENER_PAGE   = 250        # max results per screener API call


# -----------------------------
# STEP 1: GET CANDIDATE TICKERS VIA SCREENER
# Uses Yahoo Finance's own validated equity data — no stale/fake tickers.
# -----------------------------
def get_candidate_tickers():
    query = EquityQuery('and', [
        EquityQuery('eq', ['region', 'us']),
        EquityQuery('gt', ['intradaymarketcap',  MIN_MARKET_CAP]),
        EquityQuery('gt', ['avgdailyvol3m',       MIN_AVG_VOLUME]),
        EquityQuery('gt', ['intradayprice',       MIN_PRICE]),
    ])

    tickers = []
    offset  = 0

    while True:
        res = yf.screen(query, sortField='avgdailyvol3m', sortAsc=False,
                        size=SCREENER_PAGE, offset=offset)
        quotes = res.get('quotes', [])
        if not quotes:
            break
        tickers.extend(q['symbol'] for q in quotes if 'symbol' in q)
        offset += len(quotes)
        if offset >= res.get('total', 0):
            break

    return tickers


# -----------------------------
# STEP 2: TREND CHECK ON PRICE HISTORY
# -----------------------------
def is_uptrend(close: pd.Series) -> bool:
    sma20 = close.rolling(20).mean()
    if len(close) < 30 or pd.isna(sma20.iloc[-1]) or pd.isna(sma20.iloc[-5]):
        return False
    return float(close.iloc[-1]) > float(sma20.iloc[-1]) > float(sma20.iloc[-5])


# -----------------------------
# STEP 3: BATCH HISTORY DOWNLOAD + TREND FILTER
# -----------------------------
def scan_batch(batch: list) -> list:
    results = []
    try:
        raw = yf.download(
            batch,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        print(f"Batch error: {e}")
        return results

    for ticker in batch:
        try:
            if len(batch) == 1:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                lvl0 = raw.columns.get_level_values(0)
                if ticker not in lvl0:
                    continue
                df = raw[ticker].copy()

            df = df.dropna(how="all").tail(LOOKBACK)
            close = df["Close"].dropna()

            if len(close) < 30:
                continue

            if is_uptrend(close):
                avg_vol = int(df["Volume"].dropna().rolling(20).mean().iloc[-1])
                results.append({
                    "Ticker":     ticker,
                    "Last Price": round(float(close.iloc[-1]), 2),
                    "Avg Volume": avg_vol,
                })
        except Exception:
            pass

    return results


# -----------------------------
# MAIN
# -----------------------------
print("Step 1: Fetching valid US equity candidates via screener...")
candidates = get_candidate_tickers()
print(f"  {len(candidates)} candidates (vol > {MIN_AVG_VOLUME:,}, price > ${MIN_PRICE})\n")

batches = [candidates[i:i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
print(f"Step 2: Checking uptrend on {len(batches)} batches of {BATCH_SIZE}...")

results = []
for batch in tqdm(batches, unit="batch"):
    results.extend(scan_batch(batch))

# -----------------------------
# OUTPUT
# -----------------------------
df_out = pd.DataFrame(results)

if df_out.empty:
    print("\nNo stocks matched the momentum criteria.")
else:
    df_out = df_out.sort_values("Avg Volume", ascending=False).reset_index(drop=True)
    print(f"\n{len(df_out)} stocks in uptrend with sufficient liquidity:\n")
    print(df_out.to_string(index=False))
