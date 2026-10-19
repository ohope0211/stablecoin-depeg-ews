#!/usr/bin/env python3
"""
build_features.py (v3: them dex_imbalance tu Curve 3pool) - BUOC 5: tinh bien dan xuat va gan nhan depeg.

Dau vao : data/clean/panel_base.csv   (tu build_panel.py)
Dau ra  : data/clean/dataset_hourly.csv

QUY TAC CHONG LOOK-AHEAD
  - Moi BIEN (feature) tai gio t chi dung thong tin tu gio t tro ve truoc.
  - Chi NHAN (depeg_label_*) moi duoc nhin ve tuong lai.
  - supply va tvl da duoc tre 1 ngay o build_panel.py.

DINH NGHIA NHAN (nhan chinh: |price-1| > tau voi tau = 0.5%, m = 2 gio lien tiep)
  breach_t   : |price_t - 1| > tau            (ban "down": price_t - 1 < -tau, chi lech xuong)
  confirmed_t: breach_t va (breach_{t-1} hoac breach_{t+1})   -> thuoc chuoi >= 2 gio lien tiep
  onset_t    : confirmed_t va KHONG co confirmed nao trong 24 gio truoc  -> diem BAT DAU mot dot depeg
  label_h(t) = 1 neu co onset trong (t, t+h]; 0 neu cua so day du gia ma khong co onset; con lai NaN
  in_depeg_t : co breach trong 24 gio gan nhat (gom gio t). Dung de LOAI dong khi huan luyen canh bao som.
  breach_now : breach tai dung gio t (bo loc nhe hon in_depeg neu muon giu nhieu mau duong hon).

CAC BIEN THE NHAN
  depeg_label_h*       : hai phia, tau=0.5%   (NHAN CHINH)
  depeg_down_label_h*  : chi lech xuong (price < 1-tau), tau=0.5%
  depeg_tau1_label_h*  : hai phia, tau=1%     (kiem tra do ben)
  depeg_tau2_label_h*  : hai phia, tau=2%     (chi ~6 su kien: chu yeu de mo ta)

Chay:   python build_features.py
Roi dan TOAN BO output cho Claude.
"""
import os

import numpy as np
import pandas as pd

CLEAN = os.path.join("data", "clean")
HORIZONS = [1, 6, 24]
VARIANTS = [  # (hau to cot, tau, phia, tien to nhan)
    ("", 0.005, "both", "depeg_label_h"),
    ("_down", 0.005, "down", "depeg_down_label_h"),
    ("_tau1", 0.01, "both", "depeg_tau1_label_h"),
    ("_tau2", 0.02, "both", "depeg_tau2_label_h"),
]


CURVE_SHARE = {"DAI": "share_dai", "USDC": "share_usdc", "USDT": "share_usdt"}  # chi 3 coin nam trong 3pool


def load_curve():
    """Doc so du Curve 3pool (tu backfill_curve3pool.py hoac xuat tu Dune). Tra ve None neu chua co file."""
    for name in ["curve3pool_hourly.csv", "dune_curve3pool_hourly.csv"]:
        p = os.path.join("data", "raw", name)
        if os.path.exists(p):
            c = pd.read_csv(p)
            t = pd.to_datetime(c["hour"], utc=True).astype("datetime64[ns, UTC]").dt.floor("h")
            c = c.set_index(pd.DatetimeIndex(t, name="timestamp"))
            print(f"  (dex_imbalance: doc {name}, {len(c)} dong)")
            return c[~c.index.duplicated(keep="last")].sort_index()
    print("  (chua co data/raw/curve3pool_hourly.csv -> dex_imbalance de trong)")
    return None


CURVE = load_curve()


def episode_flags(price, tau, side="both"):
    """Tra ve (breach, onset, in_depeg) cho 1 chuoi gia lien tuc theo gio."""
    valid = price.notna()
    dev = price - 1
    breach = ((dev.abs() > tau) if side == "both" else (dev < -tau)) & valid
    conf = breach & (breach.shift(1, fill_value=False) | breach.shift(-1, fill_value=False))
    prior = conf.shift(1, fill_value=False).astype(float).rolling(24, min_periods=1).max()
    onset = conf & (prior == 0)
    in_depeg = breach.astype(float).rolling(24, min_periods=1).max().astype(bool)
    return breach, onset, in_depeg


def forward_label(onset, valid, h):
    """label_h(t) = 1 neu co onset trong (t, t+h].
    = 0 chi khi ca cua so (t, t+h] day du gia (khong thieu gio nao).
    = NaN neu khong co onset nhung cua so thieu gia hoac chua ket thuc (nhan chua biet).
    Neu da thay onset trong cua so thi nhan = 1 du cac gio khac co thieu gia (duong da biet chac)."""
    mx = onset.astype(float)[::-1].shift(1).rolling(h, min_periods=1).max()[::-1]
    cnt = valid.astype(float)[::-1].shift(1).rolling(h, min_periods=1).sum()[::-1]
    return mx.where((mx == 1) | (cnt == h))


def add_features(d):
    p = d["price"]
    d["peg_deviation"] = p - 1
    d["abs_dev"] = d["peg_deviation"].abs()
    d["dev_low"] = d["price_low"] - 1
    d["dev_high"] = d["price_high"] - 1
    d["range_hl"] = (d["price_high"] - d["price_low"]) / p
    d["ret_1h"] = p / p.shift(1) - 1
    d["ret_6h"] = p / p.shift(6) - 1
    d["absdev_mean_6h"] = d["abs_dev"].rolling(6, min_periods=4).mean()
    d["absdev_max_24h"] = d["abs_dev"].rolling(24, min_periods=12).max()

    vol = d["volume"]
    d["volume_24h"] = vol.rolling(24, min_periods=12).sum()
    d["liquidity"] = np.log1p(d["volume_24h"])  # thanh khoan giao dich (log volume 24h)
    lv = np.log1p(vol)
    mu = lv.rolling(720, min_periods=168).mean()
    sd = lv.rolling(720, min_periods=168).std()
    d["volume_z"] = (lv - mu) / sd.replace(0, np.nan)  # volume chuan hoa theo tung coin (30 ngay)
    d["illiq_range_24h"] = d["range_hl"].rolling(24, min_periods=12).mean()  # proxy phi thanh khoan

    d["supply"] = d["supply_usd_lag1"]
    d["supply_change"] = d["supply"] / d["supply"].shift(24) - 1
    d["supply_change_7d"] = d["supply"] / d["supply"].shift(168) - 1
    d["tvl_change"] = d["tvl_usd_lag1"] / d["tvl_usd_lag1"].shift(24) - 1
    d["btc_vol"] = np.log(d["btc_close"]).diff().rolling(24, min_periods=12).std()
    d["eth_vol"] = np.log(d["eth_close"]).diff().rolling(24, min_periods=12).std()
    # dex_imbalance (Buoc 5b): ty trong cua coin trong Curve 3pool TRU trung binh truot 30 ngay (chi dung qua khu).
    # >0 : coin dang du thua trong pool (bi ban vao pool nhieu); <0 : coin dang can dan.
    coin = d["stablecoin"].iloc[0]
    if CURVE is not None and coin in CURVE_SHARE:
        sh = CURVE[CURVE_SHARE[coin]].reindex(d.index).ffill(limit=23)  # lap gio thieu bang gia tri TRUOC DO (khong nhin tuong lai)
        d["dex_share"] = sh
        d["dex_imbalance"] = sh - sh.rolling(720, min_periods=168).mean()
    else:
        d["dex_imbalance"] = np.nan
    return d


def add_labels(d):
    valid = d["price"].notna()
    for tag, tau, side, prefix in VARIANTS:
        breach, onset, in_depeg = episode_flags(d["price"], tau, side)
        d["onset" + tag] = onset
        d["in_depeg" + tag] = in_depeg
        for h in HORIZONS:
            d[f"{prefix}{h}"] = forward_label(onset, valid, h)
        if tag == "":
            d["breach_now"] = breach
    return d


SCHEMA_FIRST = ["timestamp", "stablecoin", "price", "peg_deviation", "volume", "liquidity", "supply",
                "supply_change", "dex_imbalance", "btc_vol", "eth_vol", "tvl_change",
                "depeg_label_h1", "depeg_label_h6", "depeg_label_h24"]


def usable(ds, tag, prefix):
    return ds[(~ds["in_depeg" + tag]) & ds[f"{prefix}24"].notna()]


def main():
    path = os.path.join(CLEAN, "panel_base.csv")
    if not os.path.exists(path):
        print("Chua co data/clean/panel_base.csv, hay chay build_panel.py truoc.")
        return
    panel = pd.read_csv(path)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    out = []
    for coin, g in panel.groupby("stablecoin", sort=False):
        d = g.sort_values("timestamp").set_index("timestamp").asfreq("h")  # luoi gio lien tuc
        d["stablecoin"] = coin
        d = add_features(d)
        d = add_labels(d)
        out.append(d.reset_index())
    ds = pd.concat(out, ignore_index=True)
    rest = [c for c in ds.columns if c not in SCHEMA_FIRST]
    ds = ds[SCHEMA_FIRST + rest]
    ds.to_csv(os.path.join(CLEAN, "dataset_hourly.csv"), index=False)
    print(f"Da ghi data/clean/dataset_hourly.csv: {len(ds)} dong, {ds.shape[1]} cot")

    print("\n=== 1. NHAN CHINH (hai phia, tau=0.5%, m=2) THEO COIN ===")
    print("   'dung_duoc' = dong khong o trong depeg (in_depeg=0) va co nhan h24 hop le")
    print("   coin    tong_dong  dung_duoc  su_kien |  duong h1   duong h6   duong h24 | ty le duong h24")
    for coin, g in ds.groupby("stablecoin", sort=False):
        u = usable(g, "", "depeg_label_h")
        pos = [int((u[f"depeg_label_h{h}"] == 1).sum()) for h in HORIZONS]
        rate = pos[2] / len(u) * 100 if len(u) else float("nan")
        print(f"   {coin:6s}  {len(g):8d}  {len(u):9d}  {int(g['onset'].sum()):7d} | {pos[0]:8d}  {pos[1]:9d}  {pos[2]:10d} | {rate:6.2f}%")

    print("\n=== 2. SO SANH CAC BIEN THE NHAN (tong tat ca coin) ===")
    print("   bien_the                      su_kien  dung_duoc | duong h1  duong h6  duong h24")
    for tag, tau, side, prefix in VARIANTS:
        u = usable(ds, tag, prefix)
        pos = [int((u[f"{prefix}{h}"] == 1).sum()) for h in HORIZONS]
        name = f"{prefix}* (tau={tau * 100:g}%, {'hai phia' if side == 'both' else 'chi xuong'})"
        print(f"   {name:36s} {int(ds['onset' + tag].sum()):4d}  {len(u):9d} | {pos[0]:7d}  {pos[1]:8d}  {pos[2]:9d}")
    print("   (so dong duong h1 nho hon so su_kien vi dong ngay truoc su kien co the dang trong in_depeg)")

    print("\n=== 3. DANH SACH SU KIEN (onset) NHAN CHINH ===")
    print("   huong: TREN = gia>1 luc bat dau (premium), DUOI = gia<1 (discount)")
    print("   USDT_ref = gia USDT cung gio (de phat hien nhieu do phep nhan ty gia cheo voi BUSD/FDUSD/UST)")
    print("   thoi_diem (UTC)    coin   huong  gia_bat_dau  min_close_24h  min_low_24h  USDT_ref")
    usdt = ds[ds["stablecoin"] == "USDT"].set_index("timestamp")["price"]
    rows = []
    for coin, g in ds.groupby("stablecoin", sort=False):
        g = g.reset_index(drop=True)
        for i in g.index[g["onset"]]:
            w = g.loc[i:i + 23]
            rows.append((g.loc[i, "timestamp"], coin, g.loc[i, "price"], w["price"].min(), w["price_low"].min()))
    for ts, coin, p0, mc, ml in sorted(rows):
        ref = usdt.get(ts, np.nan)
        print(f"   {ts:%Y-%m-%d %H:%M}   {coin:6s} {'TREN' if p0 > 1 else 'DUOI':5s}  {p0:.4f}       {mc:.4f}         {ml:.4f}       {ref:.4f}")
    n_up = sum(1 for r in rows if r[2] > 1)
    print(f"   -> {len(rows)} su kien: {n_up} TREN (premium), {len(rows) - n_up} DUOI (discount)")

    print("\n=== 4. TY LE THIEU CUA CAC BIEN ===")
    cols = ["peg_deviation", "volume", "liquidity", "volume_z", "supply_change", "tvl_change",
            "btc_vol", "eth_vol", "illiq_range_24h", "dex_imbalance"]
    for coin, g in ds.groupby("stablecoin", sort=False):
        print(f"   {coin:6s} " + " | ".join(f"{c}={g[c].isna().mean() * 100:.0f}%" for c in cols))

    print("\nXong. Dan toan bo output cho Claude.")


if __name__ == "__main__":
    main()
