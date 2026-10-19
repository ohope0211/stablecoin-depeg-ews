#!/usr/bin/env python3
"""
backfill_extra.py - BUOC 3c: tai bo sung du lieu tho con thieu, kem kiem tra nhanh.

Tai:
  1. Bitstamp usdcusd, usdtusd (hourly, 2021-07 -> nay)  : nguon gia chinh cho USDC, doi chieu USDT
  2. Kraken DAIUSD, USDCUSD, USDTUSD (720 gio gan nhat)  : thu lam nguon DAI live

Chay (cung thu muc da chay backfill_raw.py):
    pip install requests pandas
    python backfill_extra.py
Roi dan TOAN BO output cho Claude.
"""
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

S = requests.Session()
S.headers.update({"User-Agent": "stablecoin-ews-student-project"})
OUT = os.path.join("data", "raw")
UTC = timezone.utc
START = datetime(2021, 7, 1, tzinfo=UTC)
NOW = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
FAILED = []


def get(url, **params):
    for attempt in range(5):
        try:
            r = S.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"   loi ({str(e)[:80]}), thu lai sau {wait}s")
            time.sleep(wait)
    return None


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f"   -> {name}: {len(df)} dong")


def bitstamp(pair):
    name = f"bitstamp_{pair}_1h.csv"
    print(f" Bitstamp {pair}")
    rows, start, end = [], int(START.timestamp()), int(NOW.timestamp())
    while start < end:
        data = get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/", step=3600, limit=1000, start=start)
        if data is None:
            FAILED.append(f"bitstamp {pair} tu {start}")
            break
        ohlc = (data.get("data") or {}).get("ohlc") or []
        if not ohlc:
            break
        rows += ohlc
        last = int(ohlc[-1]["timestamp"])
        if last + 3600 <= start:  # tranh lap vo han
            break
        start = last + 3600
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    if df.empty:
        print("   RONG")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    save(df, name)
    return name


def kraken(pair):
    name = f"kraken_{pair}_1h_recent.csv"
    print(f" Kraken {pair}")
    data = get("https://api.kraken.com/0/public/OHLC", pair=pair, interval=60)
    if not data or data.get("error"):
        print("   KHONG LAY DUOC:", (data or {}).get("error"))
        FAILED.append(f"kraken {pair}")
        return None
    res = data["result"]
    key = [k for k in res if k != "last"][0]
    df = pd.DataFrame(res[key], columns=["ts", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["timestamp"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = df[c].astype(float)
    df = df.iloc[:-1]  # bo nen dang chay do (chua dong)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "count"]]
    save(df, name)
    return name


def quick_audit(name, windows=()):
    if name is None:
        return
    df = pd.read_csv(os.path.join(OUT, name))
    t = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    s = pd.Series(df["close"].astype(float).values, index=t)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    full = pd.date_range(s.index[0], s.index[-1], freq="h")
    miss = len(full) - len(s)
    line = (f"- {name}: {len(s)} gio | {s.index[0]:%Y-%m-%d %H:%M} .. {s.index[-1]:%Y-%m-%d %H:%M} "
            f"| thieu {miss} gio ({miss / len(full) * 100:.1f}%) | min={s.min():.4f} max={s.max():.4f}")
    print(line)
    if len(s) > 1:
        hrs = (s.index[1:] - s.index[:-1]).total_seconds() / 3600 - 1
        j = int(hrs.argmax())
        if hrs[j] > 0:
            print(f"    khoang trong lon nhat: {int(hrs[j])}h, BAT DAU {s.index[j] + timedelta(hours=1):%Y-%m-%d %H:%M}")
    for a, b, label in windows:
        w = s[a:b]
        if len(w):
            print(f"    {label} {a}..{b}: min={w.min():.4f} luc {w.idxmin():%m-%d %H:%M} ({len(w)} gio)")
        else:
            print(f"    {label} {a}..{b}: KHONG CO DU LIEU")


def main():
    print("[1] Bitstamp hourly 2021-07 -> nay")
    n1 = bitstamp("usdcusd")
    n2 = bitstamp("usdtusd")
    print("\n[2] Kraken 720 gio gan nhat (thu lam nguon DAI live)")
    k = [kraken(p) for p in ["DAIUSD", "USDCUSD", "USDTUSD"]]

    print("\n=== KIEM TRA NHANH ===")
    quick_audit(n1, [("2023-03-10", "2023-03-14", "USDC su kien")])
    quick_audit(n2, [("2022-05-10", "2022-05-14", "USDT su kien")])
    for n in k:
        quick_audit(n)
    if FAILED:
        print("\nRequest that bai:")
        for x in FAILED:
            print("  -", x)
    print("\nXong. Dan toan bo output cho Claude.")


if __name__ == "__main__":
    main()
