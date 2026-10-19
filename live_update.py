#!/usr/bin/env python3
"""
live_update.py - BUOC 6: cap nhat du lieu + bien MOI GIO (chay tu dong bang GitHub Actions).

Moi lan chay:
  1. Tai ~40 ngay gan nhat tu cac nguon (chi nen da DONG, bo nen dang chay do).
  2. Lam sach / ghep nguon bang DUNG cac ham cua lich su (build_panel.py), tinh bien bang DUNG ham
     cua lich su (build_features.py) -> bien live khop voi bien huan luyen.
  3. Doc so du Curve 3pool moi nhat qua RPC va nap vao lich su Curve.
  4. Ghi:  data/live/latest.csv        (dong moi nhat cua moi coin, cho dashboard)
           data/live/features_log.csv  (nhat ky cac gio da dong, khong trung lap)
           data/live/curve3pool_live.csv

Coin live: USDT, USDC, DAI, FDUSD (BUSD ngung giao dich 2023-12, UST sap 2022 nen chi co o lich su).
Nguon: Coinbase USDT-USD | Bitstamp usdcusd | DefiLlama DAI (nhu giai doan llama_fallback o lich su)
       | Binance FDUSD/USDT x USDT/USD | Binance BTC, ETH | DefiLlama supply, TVL | Curve 3pool (RPC).
Binance: dung data-api.binance.vision (ban sao du lieu thi truong, khong bi chan theo dia ly nhu
api.binance.com khi chay tren may chu cua GitHub o My).

Chay thu tren may:   python live_update.py        (can dat ETH_RPC_URL de co dex_imbalance)
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import backfill_curve3pool as bc
import build_features as bf
import build_panel as bp

LIVE = os.path.join("data", "live")
WINDOW_H = 960  # 40 ngay: du cho cua so truot 720 gio va supply_change_7d
COINS = ["USDT", "USDC", "DAI", "FDUSD"]
SUPPLY_IDS = {"USDT": 1, "USDC": 2, "DAI": 5, "FDUSD": 119}
BINANCE_BASES = ["https://data-api.binance.vision", "https://api.binance.com"]
S = requests.Session()
S.headers.update({"User-Agent": "stablecoin-ews-student-project"})


def get(url, **params):
    for attempt in range(4):
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


def hourly_index(seconds):
    t = bp.to_ns(pd.to_datetime(pd.Series(list(seconds)).astype("int64"), unit="s", utc=True)).dt.floor("h")
    return pd.DatetimeIndex(t, name="timestamp")


def finish(df, last):
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[df.index <= pd.Timestamp(last)]  # chi nen DA DONG


# ------------------------------------------------------------------ cac nguon
def fetch_coinbase(prod, first, last):
    iso, rows, t = "%Y-%m-%dT%H:%M:%SZ", [], first
    while t <= last:
        t_end = min(t + timedelta(hours=299), last)
        data = get(f"https://api.exchange.coinbase.com/products/{prod}/candles",
                   start=t.strftime(iso), end=t_end.strftime(iso), granularity=3600)
        rows += data or []
        t += timedelta(hours=300)
        time.sleep(0.2)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    df.index = hourly_index(df.pop("ts"))
    return finish(df.astype(float), last)


def fetch_bitstamp(pair, first, last):
    data = get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/", step=3600, limit=1000, start=int(first.timestamp()))
    ohlc = ((data or {}).get("data") or {}).get("ohlc") or []
    if not ohlc:
        return None
    df = pd.DataFrame(ohlc)
    df.index = hourly_index(df.pop("timestamp"))
    return finish(df[["low", "high", "close", "volume"]].astype(float), last)


def fetch_binance(sym, first, last):
    data = None
    for base in BINANCE_BASES:
        data = get(f"{base}/api/v3/klines", symbol=sym, interval="1h",
                   startTime=int(first.timestamp() * 1000), limit=1000)
        if data:
            break
    if not data:
        return None
    df = pd.DataFrame(data).iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    df.index = hourly_index(df.pop("open_time") // 1000)
    return finish(df[["low", "high", "close", "volume", "quote_volume"]].astype(float), last)


def fetch_llama_price(key, first, last):
    rows, t = [], first
    while t <= last:
        data = get(f"https://coins.llama.fi/chart/{key}", start=int(t.timestamp()), span=168, period="1h")
        prices = (((data or {}).get("coins") or {}).get(key) or {}).get("prices") or []
        rows += [(p["timestamp"], p["price"]) for p in prices]
        t += timedelta(hours=168)
        time.sleep(0.25)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "price"])
    df.index = hourly_index(df.pop("ts"))
    return finish(df.astype(float), last)


def fetch_supply(sid):
    data = get("https://stablecoins.llama.fi/stablecoincharts/all", stablecoin=sid)
    if not data:
        return None
    rows = [(int(x["date"]), (x.get("totalCirculatingUSD") or x.get("totalCirculating") or {}).get("peggedUSD"))
            for x in data]
    df = pd.DataFrame(rows, columns=["ts", "circulating_usd"]).drop_duplicates("ts").sort_values("ts")
    df["date"] = bp.to_ns(pd.to_datetime(df.pop("ts"), unit="s", utc=True))
    return df


def fetch_tvl():
    data = get("https://api.llama.fi/v2/historicalChainTvl")
    if not data:
        return None
    df = pd.DataFrame([(int(x["date"]), x["tvl"]) for x in data], columns=["ts", "tvl_usd"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df["date"] = bp.to_ns(pd.to_datetime(df.pop("ts"), unit="s", utc=True))
    return df


# ------------------------------------------------------------------ Curve 3pool
def read_curve(path):
    if not os.path.exists(path):
        return None
    c = pd.read_csv(path)
    t = pd.to_datetime(c["hour"], utc=True).astype("datetime64[ns, UTC]").dt.floor("h")
    c = c.set_index(pd.DatetimeIndex(t, name="timestamp"))
    return c[~c.index.duplicated(keep="last")].sort_index()


def update_curve(last):
    """Them 1 dong so du Curve cho gio da dong gan nhat (so du CUOI gio do), roi tra ve lich su day du."""
    live_path = os.path.join(LIVE, "curve3pool_live.csv")
    url = os.environ.get("ETH_RPC_URL")
    if not url:
        print("  Curve: chua co ETH_RPC_URL -> khong lay duoc so du moi")
    else:
        try:
            target = int((last + timedelta(hours=1)).timestamp()) - 1
            latest = int(bc.rpc(url, [("eth_blockNumber", [])])[0], 16)
            latest_ts = bc.block_times(url, [latest])[0]
            b = bc.find_block(url, target, latest - max(0, latest_ts - target) / 12.0, latest)
            dai, usdc, usdt = bc.read_pool(url, b)
            tot = dai + usdc + usdt
            row = pd.DataFrame([{"hour": last.strftime("%Y-%m-%d %H:%M:%S"), "dai_balance": dai,
                                 "usdc_balance": usdc, "usdt_balance": usdt, "share_dai": dai / tot,
                                 "share_usdc": usdc / tot, "share_usdt": usdt / tot}])
            old = pd.read_csv(live_path) if os.path.exists(live_path) else row.iloc[0:0]
            out = pd.concat([old, row]).drop_duplicates("hour", keep="last").sort_values("hour")
            out.to_csv(live_path, index=False)
            print(f"  Curve: {last:%Y-%m-%d %H:%M} | DAI {dai / tot:.1%} USDC {usdc / tot:.1%} USDT {usdt / tot:.1%}")
        except (Exception, SystemExit) as e:  # noqa: BLE001  (loi RPC khong duoc lam hong ca pipeline)
            print(f"  Curve: LOI RPC ({str(e)[:100]}) -> bo qua gio nay, dex_imbalance dung gia tri cu")
    parts = [p for p in [read_curve(os.path.join("data", "raw", "curve3pool_hourly.csv")), read_curve(live_path)]
             if p is not None]
    if not parts:
        return None
    cur = pd.concat(parts)
    return cur[~cur.index.duplicated(keep="last")].sort_index()


# ------------------------------------------------------------------ chinh
def main():
    now = datetime.now(timezone.utc)
    last = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)  # nen DA DONG gan nhat
    first = last - timedelta(hours=WINDOW_H - 1)
    idx = pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq="h")
    print(f"Chay luc {now:%Y-%m-%d %H:%M} UTC | gio da dong moi nhat: {last:%Y-%m-%d %H:%M} | cua so {WINDOW_H}h")
    os.makedirs(LIVE, exist_ok=True)

    print("Tai du lieu...")
    cb_usdt = fetch_coinbase("USDT-USD", first, last)
    bs_usdt = fetch_bitstamp("usdtusd", first, last)
    bs_usdc = fetch_bitstamp("usdcusd", first, last)
    dai_px = fetch_llama_price("coingecko:dai", first, last)
    bn = {s: fetch_binance(f"{s}USDT", first, last) for s in ["FDUSD", "BTC", "ETH"]}
    tvl = fetch_tvl()
    supply = {c: fetch_supply(i) for c, i in SUPPLY_IDS.items()}
    for name, v in [("Coinbase USDT", cb_usdt), ("Bitstamp USDT", bs_usdt), ("Bitstamp USDC", bs_usdc),
                    ("DefiLlama DAI", dai_px), ("Binance FDUSD", bn["FDUSD"]), ("Binance BTC", bn["BTC"]),
                    ("Binance ETH", bn["ETH"]), ("TVL", tvl)]:
        print(f"  {name:14s}: {'THIEU' if v is None else str(len(v)) + ' dong'}")
    if cb_usdt is None and bs_usdt is None:
        sys.exit("Khong co gia USDT -> khong the tinh gia FDUSD, dung.")

    empty = pd.DataFrame(columns=["close", "low", "high", "volume"], dtype=float)
    usdt = bp.assemble([("coinbase", bp.clean_frame(bp.std(cb_usdt) if cb_usdt is not None else empty, "coinbase_USDT", idx)),
                        ("bitstamp", bp.clean_frame(bp.std(bs_usdt) if bs_usdt is not None else empty, "bitstamp_USDT", idx))], idx)
    usdc = bp.assemble([("bitstamp", bp.clean_frame(bp.std(bs_usdc) if bs_usdc is not None else empty, "bitstamp_USDC", idx))], idx)
    if dai_px is not None:
        llf = pd.DataFrame({"close": dai_px["price"], "low": np.nan, "high": np.nan, "volume": np.nan}, index=dai_px.index)
    else:
        llf = empty
    dai = bp.assemble([("llama_fallback", bp.clean_frame(llf, "llama_DAI", idx))], idx)
    fd = bn["FDUSD"] if bn["FDUSD"] is not None else pd.DataFrame(columns=["close", "low", "high", "quote_volume"], dtype=float)
    fdusd = bp.assemble([("binance_x_usdt", bp.cross(fd, usdt["price"], idx, "binance_FDUSD"))], idx)
    coins = {"USDT": usdt, "USDC": usdc, "DAI": dai, "FDUSD": fdusd}

    btc = bn["BTC"]["close"].reindex(idx).ffill(limit=2) if bn["BTC"] is not None else pd.Series(np.nan, index=idx)
    eth = bn["ETH"]["close"].reindex(idx).ffill(limit=2) if bn["ETH"] is not None else pd.Series(np.nan, index=idx)

    bf.CURVE = update_curve(last)  # build_features.add_features doc bien toan cuc nay

    tails = []
    print("\nTinh bien...")
    for coin, df in coins.items():
        x = df.copy()
        x.insert(0, "stablecoin", coin)
        x["supply_usd_lag1"] = bp.daily_lag(supply[coin], "circulating_usd", x.index) if supply[coin] is not None else np.nan
        x["tvl_usd_lag1"] = bp.daily_lag(tvl, "tvl_usd", x.index) if tvl is not None else np.nan
        x["btc_close"], x["eth_close"] = btc.reindex(x.index), eth.reindex(x.index)
        x.index.name = "timestamp"
        d = bf.add_features(x.asfreq("h"))
        d["stablecoin"] = coin
        t = d.tail(6)  # 6 gio gan nhat: tu va lo hong neu lan chay truoc bi tre
        tails.append(t)
        r = d.iloc[-1]
        ok = pd.notna(r["price"])
        print(f"  {coin:6s} {d.index[-1]:%H:%M} | gia={r['price']:.4f} ({r['price_source'] or 'khong co'}) "
              f"| dev={r['peg_deviation'] * 100:+.3f}% | dex_imb={r['dex_imbalance']:+.4f}" if ok else
              f"  {coin:6s} KHONG CO GIA cho gio {d.index[-1]:%H:%M}")

    new = pd.concat(tails).reset_index()
    new["timestamp"] = new["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = [c for c in bf.SCHEMA_FIRST if c in new.columns] + [c for c in new.columns if c not in bf.SCHEMA_FIRST]
    new = new[cols]
    log_path = os.path.join(LIVE, "features_log.csv")
    if os.path.exists(log_path):
        new = pd.concat([pd.read_csv(log_path), new])
    new = new.drop_duplicates(["timestamp", "stablecoin"], keep="last").sort_values(["stablecoin", "timestamp"])
    new.to_csv(log_path, index=False)
    new.groupby("stablecoin").tail(1).sort_values("stablecoin").to_csv(os.path.join(LIVE, "latest.csv"), index=False)
    print(f"\nDa ghi {LIVE}/latest.csv va features_log.csv ({len(new)} dong)")


if __name__ == "__main__":
    main()
