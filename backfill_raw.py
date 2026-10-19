#!/usr/bin/env python3
"""
backfill_raw.py - BUOC 3: tai du lieu tho lich su (raw) cho Stablecoin Depeg EWS.

Chi tai va luu raw, CHUA xu ly. Moi nguon luu 1 file CSV trong data/raw/.
Cac buoc lam sach, hop nhat, tinh bien la Buoc 4-5.

Cai dat:   pip install requests pandas
Chay het:  python backfill_raw.py
Chay tung phan:  python backfill_raw.py --only binance coinbase
   (cac phan: binance coinbase llama_prices llama_supply llama_tvl)
Tai lai file da co:  them --force

Thoi gian uoc tinh: khoang 10-20 phut (chu yeu do llama_prices).
Neu bi ngat giua chung, chay lai: file nao da xong se duoc bo qua.
"""
import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

S = requests.Session()
S.headers.update({"User-Agent": "stablecoin-ews-student-project"})

OUT = os.path.join("data", "raw")
UTC = timezone.utc
START = datetime(2021, 7, 1, tzinfo=UTC)  # moc bat dau mau hourly (xem ghi chu quyet dinh)
NOW = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
FAILED = []  # cac request that bai sau khi thu lai, in ra o cuoi


def get(url, **params):
    """GET co thu lai. Tra ve JSON hoac None neu that bai han."""
    for attempt in range(5):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"   429 (qua gioi han), doi {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"   loi ({str(e)[:80]}), thu lai sau {wait}s")
            time.sleep(wait)
    return None


def skip(name, force):
    if os.path.exists(os.path.join(OUT, name)) and not force:
        print(f"   bo qua {name} (da co, dung --force de tai lai)")
        return True
    return False


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name), index=False)
    if len(df):
        print(f"   -> {name}: {len(df)} dong, {df.iloc[0, 0]} .. {df.iloc[-1, 0]}")
    else:
        print(f"   -> {name}: RONG (kiem tra lai)")


def run_binance(force):
    print("\n[binance] klines 1h")
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    for sym in ["USDCUSDT", "BUSDUSDT", "FDUSDUSDT", "USTUSDT", "BTCUSDT", "ETHUSDT"]:
        name = f"binance_{sym}_1h.csv"
        print(f" {sym}")
        if skip(name, force):
            continue
        rows, start_ms, end_ms = [], int(START.timestamp() * 1000), int(NOW.timestamp() * 1000)
        while start_ms < end_ms:
            data = get("https://api.binance.com/api/v3/klines", symbol=sym, interval="1h",
                       startTime=start_ms, endTime=end_ms, limit=1000)
            if data is None:
                FAILED.append(f"binance {sym} tu {start_ms}")
                break
            if not data:
                break
            rows += data
            start_ms = data[-1][0] + 3600 * 1000
            time.sleep(0.1)
        df = pd.DataFrame(rows, columns=cols).drop_duplicates("open_time")
        df.insert(0, "timestamp", pd.to_datetime(df.pop("open_time"), unit="ms", utc=True))
        df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
        save(df, name)


def run_coinbase(force):
    print("\n[coinbase] candles 1h")
    iso = "%Y-%m-%dT%H:%M:%SZ"
    for prod in ["USDT-USD", "DAI-USD"]:
        name = f"coinbase_{prod}_1h.csv"
        print(f" {prod}")
        if skip(name, force):
            continue
        rows, t = [], START
        while t < NOW:
            t_end = t + timedelta(hours=299)  # 300 nen/yeu cau (gioi han cua Coinbase)
            data = get(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                       start=t.strftime(iso), end=t_end.strftime(iso), granularity=3600)
            if data is None:
                FAILED.append(f"coinbase {prod} tu {t}")
            else:
                rows += data
            t += timedelta(hours=300)
            time.sleep(0.2)
        df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
        df = df.drop_duplicates("ts").sort_values("ts")
        df.insert(0, "timestamp", pd.to_datetime(df.pop("ts"), unit="s", utc=True))
        save(df, name)


PRICE_COINS = {
    "USDT": ("coingecko:tether", START, NOW),
    "USDC": ("coingecko:usd-coin", START, NOW),
    "DAI": ("coingecko:dai", START, NOW),
    "UST": ("coingecko:terrausd", START, datetime(2022, 7, 1, tzinfo=UTC)),
    "BUSD": ("coingecko:binance-usd", START, datetime(2024, 1, 1, tzinfo=UTC)),
    "FDUSD": ("coingecko:first-digital-usd", datetime(2023, 6, 1, tzinfo=UTC), NOW),
}


def run_llama_prices(force):
    print("\n[llama_prices] gia USD theo gio tu DefiLlama coins")
    for sym, (key, t0, t1) in PRICE_COINS.items():
        name = f"llama_price_{sym}_1h.csv"
        print(f" {sym} ({key})")
        if skip(name, force):
            continue
        rows, t = [], t0
        while t < t1:
            data = get(f"https://coins.llama.fi/chart/{key}",
                       start=int(t.timestamp()), span=168, period="1h")  # 168 diem = 7 ngay
            if data is None:
                FAILED.append(f"llama_price {sym} tu {t}")
            else:
                prices = (data.get("coins", {}).get(key, {}) or {}).get("prices", [])
                rows += [(p["timestamp"], p["price"]) for p in prices]
            t += timedelta(hours=168)
            time.sleep(0.25)
        df = pd.DataFrame(rows, columns=["ts", "price"]).drop_duplicates("ts").sort_values("ts")
        df.insert(0, "timestamp", pd.to_datetime(df.pop("ts"), unit="s", utc=True))
        save(df, name)


SUPPLY_IDS = {"USDT": 1, "USDC": 2, "DAI": 5, "UST": 3, "BUSD": 4, "FDUSD": 119}


def run_llama_supply(force):
    print("\n[llama_supply] supply theo ngay (toan bo lich su)")
    for sym, sid in SUPPLY_IDS.items():
        name = f"llama_supply_{sym}_daily.csv"
        print(f" {sym} (id={sid})")
        if skip(name, force):
            continue
        data = get("https://stablecoins.llama.fi/stablecoincharts/all", stablecoin=sid)
        if data is None:
            FAILED.append(f"llama_supply {sym}")
            continue
        rows = []
        for x in data:
            circ = (x.get("totalCirculatingUSD") or x.get("totalCirculating") or {}).get("peggedUSD")
            rows.append((int(x["date"]), circ))
        df = pd.DataFrame(rows, columns=["ts", "circulating_usd"]).drop_duplicates("ts").sort_values("ts")
        df.insert(0, "date", pd.to_datetime(df.pop("ts"), unit="s", utc=True))
        save(df, name)


def run_llama_tvl(force):
    print("\n[llama_tvl] TVL toan DeFi theo ngay")
    name = "llama_tvl_all_daily.csv"
    if skip(name, force):
        return
    data = get("https://api.llama.fi/v2/historicalChainTvl")
    if data is None:
        FAILED.append("llama_tvl")
        return
    df = pd.DataFrame([(int(x["date"]), x["tvl"]) for x in data], columns=["ts", "tvl_usd"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.insert(0, "date", pd.to_datetime(df.pop("ts"), unit="s", utc=True))
    save(df, name)


RUNNERS = {
    "binance": run_binance,
    "coinbase": run_coinbase,
    "llama_supply": run_llama_supply,
    "llama_tvl": run_llama_tvl,
    "llama_prices": run_llama_prices,  # cham nhat nen de cuoi
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(RUNNERS), help="chi chay cac phan nay")
    ap.add_argument("--force", action="store_true", help="tai lai ca file da co")
    args = ap.parse_args()
    for name in (args.only or list(RUNNERS)):
        RUNNERS[name](args.force)
    print("\n=== TONG KET ===")
    if os.path.isdir(OUT):
        for f in sorted(os.listdir(OUT)):
            print(f"  {f}  ({os.path.getsize(os.path.join(OUT, f)) / 1024:.0f} KB)")
    if FAILED:
        print("\nCac request that bai (can tai lai hoac xem xet):")
        for x in FAILED:
            print("  -", x)
    else:
        print("\nKhong co request nao that bai.")


if __name__ == "__main__":
    main()
