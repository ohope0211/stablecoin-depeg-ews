#!/usr/bin/env python3
"""
audit_raw.py - BUOC 3b: kiem tra chat luong du lieu tho truoc khi sang Buoc 4.

Tra loi cau hoi: gia DefiLlama (nguon gia chinh) co du va khop voi Coinbase/Binance khong?

Chay (dung thu muc da chay backfill_raw.py, tuc la thu muc co data/raw/):
    python audit_raw.py
roi dan toan bo output cho Claude.
"""
import os

import pandas as pd

RAW = os.path.join("data", "raw")


def hourly_series(name, col):
    """Doc file raw -> Series theo gio (UTC). Tra ve (series, ty le dong lech phut)."""
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    t = pd.to_datetime(df["timestamp"], utc=True)
    off = (t.dt.minute != 0).mean()
    s = pd.Series(df[col].astype(float).values, index=t.dt.floor("h"))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s, off


def summarize(name, col, stable):
    s, off = hourly_series(name, col)
    print(f"\n- {name}")
    if s is None:
        print("   THIEU FILE")
        return None
    full = pd.date_range(s.index[0], s.index[-1], freq="h")
    miss = len(full) - len(s)
    print(f"   {len(s)} gio co du lieu | {s.index[0]:%Y-%m-%d} .. {s.index[-1]:%Y-%m-%d %H:%M} | "
          f"thieu {miss} gio ({miss / max(len(full), 1) * 100:.1f}%) | lech phut: {off * 100:.0f}% dong")
    if len(s) > 1:
        gaps = s.index.to_series().diff().dt.total_seconds() / 3600 - 1
        big = gaps[gaps >= 6].sort_values(ascending=False).head(5)
        if len(big):
            print("   khoang trong lon nhat: " +
                  "; ".join(f"{i:%Y-%m-%d}({int(v)}h)" for i, v in big.items()))
    if stable:
        dev = (s - 1).abs()
        print(f"   gia min={s.min():.4f} max={s.max():.4f} | so gio |lech|>0.5%: {(dev > 0.005).sum()} "
              f"| >1%: {(dev > 0.01).sum()} | >2%: {(dev > 0.02).sum()}")
    return s


def compare(a, b, label):
    if a is None or b is None:
        print(f"  {label}: thieu du lieu")
        return
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if j.empty:
        print(f"  {label}: khong co gio chung")
        return
    d = (j.a - j.b).abs()
    corr = (j.a - 1).corr(j.b - 1)
    print(f"  {label}: {len(j)} gio chung | corr(do lech neo)={corr:.3f} | mean|diff|={d.mean():.5f} "
          f"| p99={d.quantile(.99):.5f} | max={d.max():.5f} ({d.idxmax():%Y-%m-%d %H:%M}) "
          f"| %gio diff>0.001: {(d > 0.001).mean() * 100:.1f}%")


def wmin(s, a, b):
    if s is None:
        return "thieu file"
    w = s[a:b]
    if len(w) == 0:
        return "khong co du lieu"
    return f"min={w.min():.4f} luc {w.idxmin():%m-%d %H:%M} ({len(w)} gio co du lieu)"


def main():
    print("=== A. TOM TAT TUNG FILE THEO GIO ===")
    stable_files = [("llama_price_USDT_1h.csv", "price"), ("llama_price_USDC_1h.csv", "price"),
                    ("llama_price_DAI_1h.csv", "price"), ("llama_price_UST_1h.csv", "price"),
                    ("llama_price_BUSD_1h.csv", "price"), ("llama_price_FDUSD_1h.csv", "price"),
                    ("coinbase_USDT-USD_1h.csv", "close"), ("coinbase_DAI-USD_1h.csv", "close"),
                    ("binance_USDCUSDT_1h.csv", "close"), ("binance_BUSDUSDT_1h.csv", "close"),
                    ("binance_FDUSDUSDT_1h.csv", "close"), ("binance_USTUSDT_1h.csv", "close")]
    S = {}
    for name, col in stable_files:
        S[name] = summarize(name, col, stable=True)
    for name in ["binance_BTCUSDT_1h.csv", "binance_ETHUSDT_1h.csv"]:
        summarize(name, "close", stable=False)

    ll = {k: S.get(f"llama_price_{k}_1h.csv") for k in ["USDT", "USDC", "DAI", "UST", "BUSD", "FDUSD"]}
    cb_usdt = S.get("coinbase_USDT-USD_1h.csv")
    cb_dai = S.get("coinbase_DAI-USD_1h.csv")
    bn = {k: S.get(f"binance_{k}USDT_1h.csv") for k in ["USDC", "BUSD", "FDUSD", "UST"]}

    print("\n=== B. DOI CHIEU GIA DEFILLAMA VOI SAN ===")
    compare(ll["USDT"], cb_usdt, "USDT: DefiLlama vs Coinbase USDT-USD")
    compare(ll["DAI"], cb_dai, "DAI : DefiLlama vs Coinbase DAI-USD")
    if bn["USDC"] is not None and cb_usdt is not None:
        usdc_cross = (bn["USDC"] * cb_usdt).dropna()
        compare(ll["USDC"], usdc_cross, "USDC: DefiLlama vs Binance USDCUSDT x Coinbase USDT-USD")
        for k in ["BUSD", "FDUSD", "UST"]:
            if bn[k] is not None:
                compare(ll[k], (bn[k] * cb_usdt).dropna(), f"{k}: DefiLlama vs Binance {k}USDT x USDT-USD")

    print("\n=== C. CAC SU KIEN KHUNG HOANG DA BI BAT CHUA? ===")
    ust_cross = (bn["UST"] * cb_usdt).dropna() if bn["UST"] is not None and cb_usdt is not None else None
    usdc_cross = (bn["USDC"] * cb_usdt).dropna() if bn["USDC"] is not None and cb_usdt is not None else None
    busd_cross = (bn["BUSD"] * cb_usdt).dropna() if bn["BUSD"] is not None and cb_usdt is not None else None
    print("UST  2022-05-07..05-14")
    print("   DefiLlama      :", wmin(ll["UST"], "2022-05-07", "2022-05-14"))
    print("   Binance x USDT :", wmin(ust_cross, "2022-05-07", "2022-05-14"))
    print("USDC 2023-03-10..03-14")
    print("   DefiLlama      :", wmin(ll["USDC"], "2023-03-10", "2023-03-14"))
    print("   Binance x USDT :", wmin(usdc_cross, "2023-03-10", "2023-03-14"))
    print("DAI  2023-03-10..03-14")
    print("   DefiLlama      :", wmin(ll["DAI"], "2023-03-10", "2023-03-14"))
    print("   Coinbase       :", wmin(cb_dai, "2023-03-10", "2023-03-14"))
    print("USDT 2022-05-10..05-14")
    print("   DefiLlama      :", wmin(ll["USDT"], "2022-05-10", "2022-05-14"))
    print("   Coinbase       :", wmin(cb_usdt, "2022-05-10", "2022-05-14"))
    print("BUSD 2023-02-12..02-20 (Paxos bi yeu cau ngung mint)")
    print("   DefiLlama      :", wmin(ll["BUSD"], "2023-02-12", "2023-02-20"))
    print("   Binance x USDT :", wmin(busd_cross, "2023-02-12", "2023-02-20"))

    print("\nXong. Dan toan bo output cho Claude.")


if __name__ == "__main__":
    main()
