#!/usr/bin/env python3
"""
build_panel.py - BUOC 4: lam sach va hop nhat du lieu tho thanh panel hourly.

Dau vao : data/raw/*.csv (tu backfill_raw.py va backfill_extra.py)
Dau ra  : data/clean/panel_base.csv      (panel dang long: timestamp x stablecoin)
          data/clean/cleaning_log.csv    (moi diem bi loai/sua va ly do)
Ban chua tinh bien dan xuat va nhan (peg_deviation, supply_change, label...): do la BUOC 5.

Nguon gia (da chot sau khi kiem tra):
  USDT : Coinbase USDT-USD, lap lo hong bang Bitstamp usdtusd
  USDC : Bitstamp usdcusd
  DAI  : Coinbase DAI-USD (den 2026-05-04); sau do CoinGecko (neu co data/raw/coingecko_dai_1h.csv),
         neu khong co thi DefiLlama (price_source='llama_fallback', chat luong thap hon)
         Kraken DAI bi loai vi nhieu (thanh khoan mong), chi dung de doi chieu
  BUSD, FDUSD, UST : Binance X/USDT nhan voi gia USDT/USD
Gia DefiLlama KHONG dung lam gia chinh, chi dung doi chieu.
Lam sach: chi loai gia NGOAI khoang 0.5-1.5 (tick vo ly, tru UST). Cac "nhay don le" >5% chi duoc
GHI LAI, KHONG xoa, vi co the la depeg that phuc hoi nhanh (vd DAI 2023-03-11).

Chay:   pip install pandas numpy
        python build_panel.py
Roi dan TOAN BO output cho Claude.
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
CLEAN = os.path.join("data", "clean")
LOG = []  # moi phan tu: dict(timestamp, source, reason, value)


def to_ns(s):
    return s.astype("datetime64[ns, UTC]")


def read_hourly(name, optional=False):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        if not optional:
            print(f"  THIEU FILE: {name}")
        return None
    df = pd.read_csv(p)
    t = to_ns(pd.to_datetime(df["timestamp"], utc=True)).dt.floor("h")
    df = df.set_index(pd.DatetimeIndex(t, name="timestamp"))
    return df[~df.index.duplicated(keep="last")].sort_index()


def read_daily(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        print(f"  THIEU FILE: {name}")
        return None
    df = pd.read_csv(p)
    df["date"] = to_ns(pd.to_datetime(df["date"], utc=True))
    return df.sort_values("date")


def std(df, close="close", low="low", high="high", volume="volume"):
    return pd.DataFrame({"close": df[close], "low": df[low], "high": df[high], "volume": df[volume]})


def clean_frame(df, label, idx, band=True):
    """Dua ve luoi gio chung, loai tick loi. Moi diem bi loai duoc ghi vao LOG."""
    d = df.reindex(idx).astype(float)
    if band:  # ngoai [0.5, 1.5] la sai ro rang (tru UST)
        bad = (d.close < 0.5) | (d.close > 1.5)
        for ts, v in d.close[bad].items():
            LOG.append(dict(timestamp=ts, source=label, reason="ngoai_bien_0.5_1.5", value=v))
        d.loc[bad, ["close", "low", "high"]] = np.nan
        d.loc[(d.low < 0.5) | (d.low > 1.5), "low"] = np.nan
        d.loc[(d.high < 0.5) | (d.high > 1.5), "high"] = np.nan
    # nhay don le: lech >5% so voi trung binh 2 gio ke ben, trong khi 2 gio ke ben khop nhau (<1%).
    # CHI GHI LAI, KHONG XOA: day cung la hinh dang cua mot cu depeg that phuc hoi nhanh.
    prev, nxt = d.close.shift(1), d.close.shift(-1)
    spike = (((d.close - (prev + nxt) / 2).abs() > 0.05) & ((prev - nxt).abs() < 0.01)).fillna(False)
    for ts, v in d.close[spike].items():
        LOG.append(dict(timestamp=ts, source=label, reason="nhay_don_le_GIU_LAI", value=v))
    d.loc[d.close.isna(), ["low", "high"]] = np.nan
    d["low"] = pd.concat([d.low.fillna(d.close), d.close], axis=1).min(axis=1)
    d["high"] = pd.concat([d.high.fillna(d.close), d.close], axis=1).max(axis=1)
    return d


def assemble(parts, idx):
    """Ghep nhieu nguon theo thu tu uu tien; nguon sau chi lap cho nguon truoc con trong."""
    out = pd.DataFrame({"price": np.nan, "price_low": np.nan, "price_high": np.nan, "volume": np.nan}, index=idx)
    out["price_source"] = ""
    for label, d in parts:
        take = out["price"].isna() & d["close"].notna()
        out.loc[take, "price"] = d.loc[take, "close"]
        out.loc[take, "price_low"] = d.loc[take, "low"]
        out.loc[take, "price_high"] = d.loc[take, "high"]
        out.loc[take, "volume"] = d.loc[take, "volume"]
        out.loc[take, "price_source"] = label
    ff = out["price"].ffill(limit=2)  # chi lap lo hong toi da 2 gio
    fill = out["price"].isna() & ff.notna()
    for c in ["price", "price_low", "price_high"]:
        out.loc[fill, c] = ff[fill]
    out.loc[fill, "price_source"] = "ffill"
    return out


def cross(sym_df, usdt_price, idx, label, band=True):
    """Gia X/USDT (Binance) nhan USDT/USD -> gia X/USD. Volume = quote volume (USDT ~ USD)."""
    s = sym_df.reindex(idx)[["close", "low", "high", "quote_volume"]].astype(float)
    u = usdt_price
    raw = pd.DataFrame({"close": s["close"] * u, "low": s["low"] * u, "high": s["high"] * u,
                        "volume": s["quote_volume"]})
    return clean_frame(raw, label, idx, band=band)


def daily_lag(daily, value_col, times):
    """Gia tri theo ngay D chi duoc dung tu 00:00 ngay D+1 (tranh look-ahead)."""
    d = daily[["date", value_col]].copy()
    d["avail"] = d["date"] + pd.Timedelta(days=1)
    left = pd.DataFrame({"timestamp": to_ns(pd.Series(times, dtype="datetime64[ns, UTC]"))})
    m = pd.merge_asof(left, d[["avail", value_col]].sort_values("avail"), left_on="timestamp",
                      right_on="avail", direction="backward", tolerance=pd.Timedelta(days=3))
    return m[value_col].values


def longest_nan_run(s):
    isn = s.isna()
    if not isn.any():
        return 0, None
    grp = (~isn).cumsum()
    runs = isn.groupby(grp).sum()
    g = runs.idxmax()
    return int(runs.max()), s.index[(grp == g) & isn][0]


def episodes(s, tau):
    flag = ((s - 1).abs() > tau).fillna(False)
    conf = flag & (flag.shift(1, fill_value=False) | flag.shift(-1, fill_value=False))
    starts = conf & ~conf.shift(1, fill_value=False)
    return int(starts.sum()), int(flag.sum())


def compare(a, b, label):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if j.empty:
        print(f"  {label}: khong co gio chung")
        return
    d = (j.a - j.b).abs()
    corr = (j.a - 1).corr(j.b - 1)
    print(f"  {label}: {len(j)} gio chung | corr(do lech neo)={corr:.3f} | mean|diff|={d.mean():.5f} "
          f"| p99={d.quantile(.99):.5f} | max={d.max():.5f} ({d.idxmax():%Y-%m-%d %H:%M})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01", help="moc bat dau panel (UTC)")
    args = ap.parse_args()

    print("Doc du lieu tho...")
    cb_usdt = read_hourly("coinbase_USDT-USD_1h.csv")
    cb_dai = read_hourly("coinbase_DAI-USD_1h.csv")
    bs_usdt = read_hourly("bitstamp_usdtusd_1h.csv")
    bs_usdc = read_hourly("bitstamp_usdcusd_1h.csv")
    kr = {p: read_hourly(f"kraken_{p}_1h_recent.csv") for p in ["DAIUSD", "USDCUSD", "USDTUSD"]}
    bn = {s: read_hourly(f"binance_{s}USDT_1h.csv") for s in ["USDC", "BUSD", "FDUSD", "UST", "BTC", "ETH"]}
    llama_dai = read_hourly("llama_price_DAI_1h.csv")
    cg_dai = read_hourly("coingecko_dai_1h.csv", optional=True)
    tvl = read_daily("llama_tvl_all_daily.csv")
    supply = {c: read_daily(f"llama_supply_{c}_daily.csv") for c in ["USDT", "USDC", "DAI", "UST", "BUSD", "FDUSD"]}

    need = [cb_usdt, bs_usdt, bs_usdc, cb_dai, bn["BUSD"], bn["FDUSD"], bn["UST"], bn["BTC"], bn["ETH"]]
    if any(x is None for x in need):
        print("Thieu file quan trong o tren, hay chay lai backfill_raw.py / backfill_extra.py.")
        return

    frames = [f for f in [cb_usdt, cb_dai, bs_usdt, bs_usdc, *kr.values(), *bn.values()] if f is not None]
    end = max(f.index.max() for f in frames)
    idx = pd.date_range(pd.Timestamp(args.start, tz="UTC"), end, freq="h")
    print(f"Luoi gio chung: {idx[0]} .. {idx[-1]} ({len(idx)} gio)")

    # ---- gia USDT/USD (nen de quy doi cho BUSD, FDUSD, UST)
    c_cb_usdt = clean_frame(std(cb_usdt), "coinbase_USDT", idx)
    c_bs_usdt = clean_frame(std(bs_usdt), "bitstamp_USDT", idx)
    usdt = assemble([("coinbase", c_cb_usdt), ("bitstamp", c_bs_usdt)], idx)

    # ---- USDC
    c_bs_usdc = clean_frame(std(bs_usdc), "bitstamp_USDC", idx)
    usdc = assemble([("bitstamp", c_bs_usdc)], idx)
    c_bn_usdc = cross(bn["USDC"], usdt["price"], idx, "binance_USDC_cross")  # chi de doi chieu

    # ---- DAI
    c_cb_dai = clean_frame(std(cb_dai), "coinbase_DAI", idx)
    parts = [("coinbase", c_cb_dai)]
    if cg_dai is not None:
        cg = pd.DataFrame({"close": cg_dai["price"], "low": np.nan, "high": np.nan, "volume": np.nan},
                          index=cg_dai.index)
        parts.append(("coingecko", clean_frame(cg, "coingecko_DAI", idx)))
    elif llama_dai is not None:
        ll = llama_dai[llama_dai.index > cb_dai.index.max()]  # chi dung SAU khi Coinbase ket thuc
        llf = pd.DataFrame({"close": ll["price"].astype(float), "low": np.nan, "high": np.nan,
                            "volume": np.nan}, index=ll.index)
        parts.append(("llama_fallback", clean_frame(llf, "llama_DAI", idx)))
        print("  (DAI sau 2026-05-04 lay tu DefiLlama: gan nhan price_source='llama_fallback', chat luong thap hon)")
    else:
        print("  (khong co nguon DAI sau 2026-05-04: DAI se de trong)")
    dai = assemble(parts, idx)
    c_kr_dai = clean_frame(std(kr["DAIUSD"]), "kraken_DAI", idx) if kr["DAIUSD"] is not None else None

    # ---- BUSD, FDUSD, UST (Binance x USDT/USD)
    busd = assemble([("binance_x_usdt", cross(bn["BUSD"], usdt["price"], idx, "binance_BUSD"))], idx)
    fdusd = assemble([("binance_x_usdt", cross(bn["FDUSD"], usdt["price"], idx, "binance_FDUSD"))], idx)
    ust = assemble([("binance_x_usdt", cross(bn["UST"], usdt["price"], idx, "binance_UST", band=False))], idx)

    coins = {"USDT": usdt, "USDC": usdc, "DAI": dai, "BUSD": busd, "FDUSD": fdusd, "UST": ust}

    # ---- BTC, ETH
    btc = bn["BTC"]["close"].reindex(idx).ffill(limit=2)
    eth = bn["ETH"]["close"].reindex(idx).ffill(limit=2)

    # ---- hop nhat thanh panel dang long
    out = []
    for coin, df in coins.items():
        first, last = df["price"].first_valid_index(), df["price"].last_valid_index()
        if first is None:
            print(f"  {coin}: khong co gia nao!")
            continue
        x = df.loc[first:last].copy()
        x.insert(0, "stablecoin", coin)
        if supply[coin] is not None:
            x["supply_usd_lag1"] = daily_lag(supply[coin], "circulating_usd", x.index)
        else:
            x["supply_usd_lag1"] = np.nan
        x["tvl_usd_lag1"] = daily_lag(tvl, "tvl_usd", x.index) if tvl is not None else np.nan
        x["btc_close"] = btc.reindex(x.index)
        x["eth_close"] = eth.reindex(x.index)
        x.index.name = "timestamp"
        out.append(x.reset_index())
    panel = pd.concat(out, ignore_index=True)
    os.makedirs(CLEAN, exist_ok=True)
    panel.to_csv(os.path.join(CLEAN, "panel_base.csv"), index=False)
    pd.DataFrame(LOG).to_csv(os.path.join(CLEAN, "cleaning_log.csv"), index=False)

    # ================= BAO CAO =================
    print("\n=== 1. TOM TAT PANEL THEO COIN ===")
    for coin, df in coins.items():
        x = df.loc[df["price"].first_valid_index():df["price"].last_valid_index()]
        n_nan = int(x["price"].isna().sum())
        run, start = longest_nan_run(x["price"])
        src = x["price_source"].value_counts().to_dict()
        print(f"- {coin}: {len(x)} gio | {x.index[0]:%Y-%m-%d} .. {x.index[-1]:%Y-%m-%d %H:%M} "
              f"| thieu gia {n_nan} gio ({n_nan / len(x) * 100:.1f}%)")
        if run:
            print(f"    khoang trong lon nhat: {run}h, bat dau {start:%Y-%m-%d %H:%M}")
        print(f"    nguon gia: {src}")
        print(f"    gia min={x['price'].min():.4f} max={x['price'].max():.4f}")
        sup_nan = panel.loc[panel.stablecoin == coin, "supply_usd_lag1"].isna().mean() * 100
        print(f"    supply thieu: {sup_nan:.1f}% dong")

    print("\n=== 2. NHAT KY LAM SACH (loai = ngoai_bien; GIU_LAI = chi ghi nhan, khong xoa) ===")
    if LOG:
        cnt = Counter((r["source"], r["reason"]) for r in LOG)
        for (s, r), n in sorted(cnt.items()):
            print(f"  {s:22s} {r:22s} {n} diem")
        print("  20 diem dang chu y nhat (lech xa 1.0):")
        for r in sorted(LOG, key=lambda r: -abs(r["value"] - 1))[:20]:
            print(f"    {r['timestamp']:%Y-%m-%d %H:%M}  {r['source']:22s} {r['reason']:22s} gia={r['value']:.4f}")
    else:
        print("  Khong co diem nao bi loai.")

    print("\n=== 3. DOI CHIEU CHEO GIUA CAC NGUON ===")
    compare(c_bs_usdt["close"], c_cb_usdt["close"], "USDT: Bitstamp vs Coinbase")
    compare(c_bs_usdc["close"], c_bn_usdc["close"], "USDC: Bitstamp vs Binance x USDT/USD")
    if kr["USDTUSD"] is not None:
        compare(clean_frame(std(kr["USDTUSD"]), "kraken_USDT", idx)["close"], c_cb_usdt["close"],
                "USDT: Kraken vs Coinbase (30 ngay gan nhat)")
    if kr["USDCUSD"] is not None:
        compare(clean_frame(std(kr["USDCUSD"]), "kraken_USDC", idx)["close"], c_bs_usdc["close"],
                "USDC: Kraken vs Bitstamp (30 ngay gan nhat)")
    if llama_dai is not None:
        compare(llama_dai["price"].astype(float).loc["2026-01-01":"2026-05-04"], c_cb_dai["close"],
                "DAI: DefiLlama vs Coinbase (2026-01..05, giai doan yen)")
    if c_kr_dai is not None and llama_dai is not None:
        k = c_kr_dai["close"].dropna()
        ld = llama_dai["price"].astype(float).reindex(idx)
        compare(c_kr_dai["close"], ld, "DAI: Kraken vs DefiLlama (30 ngay gan nhat)")
        w = slice(k.index[0], k.index[-1])
        print(f"    30 ngay do: Kraken DAI min={k.min():.4f} max={k.max():.4f} | "
              f"DefiLlama DAI min={ld[w].min():.4f} max={ld[w].max():.4f}")

    print("\n=== 4. UOC LUONG SO SU KIEN DEPEG (|lech|>tau it nhat 2 gio lien tiep) ===")
    print("   coin    gio_hoat_dong |  tau=0.5%: su_kien/gio  |  tau=1%: su_kien/gio  |  tau=2%: su_kien/gio")
    for coin, df in coins.items():
        x = df["price"].loc[df["price"].first_valid_index():df["price"].last_valid_index()]
        cells = []
        for tau in [0.005, 0.01, 0.02]:
            e, h = episodes(x, tau)
            cells.append(f"{e:4d} / {h:5d}")
        print(f"   {coin:6s}  {len(x):8d}      |  {cells[0]}  |  {cells[1]}  |  {cells[2]}")

    print(f"\nDa ghi: {os.path.join(CLEAN, 'panel_base.csv')} ({len(panel)} dong) va cleaning_log.csv")
    print("Xong. Dan toan bo output cho Claude.")


if __name__ == "__main__":
    main()
