import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
MIN_AVG_VOLUME = 500_000     # liquidity filter
LOOKBACK = 120               # history window
BATCH_SIZE = 100             # tickers per yfinance bulk download call
MIN_PRICE = 1.0              # filter out penny stocks


# -----------------------------
# FETCH ALL US TICKERS
# -----------------------------
def fetch_us_tickers():
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    tickers = set()
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), sep="|")
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] == "N"]
            col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            raw = df[col].dropna().astype(str).tolist()
            valid = [t.strip() for t in raw if t.isalpha() and 1 <= len(t) <= 5]
            tickers.update(valid)
        except Exception as e:
            print(f"Warning: could not fetch {url}: {e}")
    return sorted(tickers)


# -----------------------------
# TREND / VOLUME CHECKS
# -----------------------------
def is_uptrend(close):
    sma20 = close.rolling(20).mean()
    if len(close) < 30 or pd.isna(sma20.iloc[-1]):
        return False
    return float(close.iloc[-1]) > float(sma20.iloc[-1]) > float(sma20.iloc[-5])


def has_volume(volume):
    avg = float(volume.rolling(20).mean().iloc[-1])
    return avg > MIN_AVG_VOLUME and float(volume.iloc[-1]) > 0.5 * avg


# -----------------------------
# BATCH SCANNER
# -----------------------------
def scan_batch(batch):
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
        print(f"  Batch download error: {e}")
        return results

    for ticker in batch:
        try:
            if len(batch) == 1:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].copy()

            df = df.dropna(how="all").tail(LOOKBACK)
            if df.empty or len(df) < 30:
                continue

            close  = df["Close"].dropna()
            volume = df["Volume"].dropna()

            if float(close.iloc[-1]) < MIN_PRICE:
                continue

            if is_uptrend(close) and has_volume(volume):
                results.append({
                    "Ticker":     ticker,
                    "Last Price": round(float(close.iloc[-1]), 2),
                    "Avg Volume": int(volume.rolling(20).mean().iloc[-1]),
                    "Status":     "PASS",
                })
        except Exception:
            pass

    return results


# -----------------------------
# RUN SCREENER
# -----------------------------
print("Fetching US ticker list...")
tickers = fetch_us_tickers()

batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
print(f"Scanning {len(tickers)} tickers in {len(batches)} batches of {BATCH_SIZE}...")

results = []
for batch in tqdm(batches, unit="batch"):
    results.extend(scan_batch(batch))

# -----------------------------
# OUTPUT
# -----------------------------
df_results = pd.DataFrame(results)

if df_results.empty:
    print("\nNo stocks matched your criteria.")
else:
    df_results = df_results.sort_values("Avg Volume", ascending=False).reset_index(drop=True)
    print(f"\n{len(df_results)} stocks passed the screener:\n")
    print(df_results.to_string(index=False))
