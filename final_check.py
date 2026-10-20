#!/usr/bin/env python3
"""
final_check.py - BUOC 7: kiem tra cuoi dataset_hourly.csv va pipeline live.

Kiem tra:
  1. Du 15 cot schema bat buoc, khong trung (timestamp, stablecoin), thoi gian tang dan.
  2. Nhan hop le (0/1/NaN) va nhat quan: h1=1 thi h6=1 thi h24=1.
  3. KHONG LOOK-AHEAD: tinh lai bien tren panel bi CAT tai moc T, so voi ban day du; moi bien tai gio <= T
     phai GIONG HET. (Neu bien dung thong tin tuong lai thi se lech.)
  4. Doi chieu data/live/latest.csv: cot khop schema, gia hop ly; so khop voi lich su neu co gio trung.
Chay:   python final_check.py        (dung thu muc co data/clean/dataset_hourly.csv va panel_base.csv)
Roi dan TOAN BO output cho Claude.
"""
import os

import numpy as np
import pandas as pd

import build_features as bf

CLEAN = os.path.join("data", "clean")
LIVE = os.path.join("data", "live")
SCHEMA = ["timestamp", "stablecoin", "price", "peg_deviation", "volume", "liquidity", "supply", "supply_change",
          "dex_imbalance", "btc_vol", "eth_vol", "tvl_change", "depeg_label_h1", "depeg_label_h6", "depeg_label_h24"]
LABELS = ["depeg_label_h1", "depeg_label_h6", "depeg_label_h24"]
cnt = {"ok": 0, "loi": 0}


def report(ok, text):
    cnt["ok" if ok else "loi"] += 1
    print(f"   [{'DAT ' if ok else 'LOI '}] {text}")


def main():
    p = os.path.join(CLEAN, "dataset_hourly.csv")
    if not os.path.exists(p):
        print("Chua co", p)
        return
    ds = pd.read_csv(p)
    ds["timestamp"] = pd.to_datetime(ds["timestamp"], utc=True)
    print(f"Dataset: {len(ds)} dong, {ds.shape[1]} cot, {ds['stablecoin'].nunique()} coin\n")

    print("=== 1. CAU TRUC ===")
    miss = [c for c in SCHEMA if c not in ds.columns]
    report(not miss, "du 15 cot schema bat buoc" + (f" (thieu: {miss})" if miss else ""))
    dup = int(ds.duplicated(["timestamp", "stablecoin"]).sum())
    report(dup == 0, f"khong co dong trung (timestamp, stablecoin): {dup} dong trung")
    mono = all(g["timestamp"].is_monotonic_increasing for _, g in ds.groupby("stablecoin"))
    report(mono, "timestamp tang dan trong tung coin")
    bad = ds[(ds["price"] < 0.5) | (ds["price"] > 1.5)]
    by = bad.groupby("stablecoin").size().to_dict()
    other = {k: v for k, v in by.items() if k != "UST"}
    report(not other, f"gia nam trong [0.5, 1.5] (tru UST, vi UST sup do that va khong ap bo loc): ngoai khoang theo coin = {by or 0}")
    if "UST" in by:
        u = bad[bad["stablecoin"] == "UST"]
        print(f"      UST ngoai khoang: {len(u)} dong, tu {u['timestamp'].min():%Y-%m-%d %H:%M} den {u['timestamp'].max():%Y-%m-%d %H:%M}, gia min = {u['price'].min():.4f}")
    for coin, g in ds.groupby("stablecoin", sort=False):
        gaps = g["price"].isna().mean() * 100
        print(f"      {coin:6s} {g['timestamp'].min():%Y-%m-%d} .. {g['timestamp'].max():%Y-%m-%d} | thieu gia {gaps:.1f}%")

    print("\n=== 2. NHAN ===")
    vals = pd.unique(ds[LABELS].values.ravel())
    report(all((pd.isna(v) or v in (0, 1)) for v in vals), "nhan chi gom 0, 1, NaN")
    v = ds.dropna(subset=LABELS)
    inc = ((v["depeg_label_h1"] == 1) & (v["depeg_label_h6"] == 0)).sum() + \
          ((v["depeg_label_h6"] == 1) & (v["depeg_label_h24"] == 0)).sum()
    report(inc == 0, f"nhat quan h1 <= h6 <= h24: {int(inc)} dong mau thuan")
    last = ds.sort_values("timestamp").groupby("stablecoin").tail(1)
    report(last["depeg_label_h24"].isna().all() or (last["depeg_label_h24"] == 1).any(),
           "dong cuoi cung cua moi coin chua co nhan h24 (tuong lai chua biet)")

    print("\n=== 3. KIEM TRA LOOK-AHEAD (cat panel tai moc T roi tinh lai bien) ===")
    pp = os.path.join(CLEAN, "panel_base.csv")
    if not os.path.exists(pp):
        print("   (khong co panel_base.csv, bo qua)")
    else:
        panel = pd.read_csv(pp)
        panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
        feats = ["peg_deviation", "abs_dev", "range_hl", "ret_1h", "ret_6h", "absdev_mean_6h", "absdev_max_24h",
                 "volume_24h", "liquidity", "volume_z", "illiq_range_24h", "supply", "supply_change",
                 "supply_change_7d", "tvl_change", "btc_vol", "eth_vol", "dex_imbalance", "illiq_med_30d", "tau_dyn"]
        tmin, tmax = panel["timestamp"].min(), panel["timestamp"].max()
        cutoffs = [pd.Timestamp(x, tz="UTC") for x in ("2022-05-12 00:00", "2023-03-11 00:00", "2025-04-02 12:00")
                   if tmin + pd.Timedelta(days=60) < pd.Timestamp(x, tz="UTC") < tmax]
        if len(cutoffs) < 3:  # du lieu ngan: dung moc chia theo ty le
            cutoffs = [tmin + (tmax - tmin) * f for f in (0.35, 0.6, 0.85)]
        for T in cutoffs:
            T = T.floor("h")
            worst, worst_col, n, used = 0.0, "", 0, []
            for coin, g in panel.groupby("stablecoin", sort=False):
                g = g.sort_values("timestamp").set_index("timestamp").asfreq("h")
                g["stablecoin"] = coin
                part = g.loc[:T]
                if len(part) < 300:  # coin chua ton tai (hoac moi co qua it du lieu) tai moc cat nay
                    continue
                used.append(coin)
                full = bf.add_features(g.copy())
                cut = bf.add_features(part.copy())
                a, b = full.loc[:T, feats], cut[feats]
                diff = (a - b).abs()
                same_nan = a.isna() == b.isna()
                n += int(same_nan.size)
                if not same_nan.all().all():
                    worst, worst_col = np.inf, f"{coin}: NaN khac nhau"
                m = np.nanmax(diff.values) if diff.notna().any().any() else 0.0
                if m > worst:
                    worst, worst_col = m, f"{coin}"
            report(worst < 1e-9, f"cat tai {T:%Y-%m-%d %H:%M}: do lech lon nhat = {worst:.2e} ({n} o so sanh; coin: {','.join(used)})"
                   + ("" if worst < 1e-9 else f" -> KIEM TRA {worst_col}"))

    print("\n=== 4. PIPELINE LIVE ===")
    lp = os.path.join(LIVE, "latest.csv")
    if not os.path.exists(lp):
        print("   (chua co data/live/latest.csv)")
    else:
        lv = pd.read_csv(lp)
        miss = [c for c in SCHEMA if c not in lv.columns and not c.startswith("depeg_label")]
        report(not miss, "latest.csv du cac cot bien (tru nhan, vi nhan chi biet sau)" + (f" thieu: {miss}" if miss else ""))
        cols = ["stablecoin", "timestamp", "price", "peg_deviation", "volume", "supply", "dex_imbalance", "btc_vol"]
        print(lv[[c for c in cols if c in lv.columns]].to_string(index=False))
        lg = os.path.join(LIVE, "features_log.csv")
        if os.path.exists(lg):
            log = pd.read_csv(lg)
            log["timestamp"] = pd.to_datetime(log["timestamp"], utc=True)
            m = ds.merge(log, on=["timestamp", "stablecoin"], suffixes=("_hist", "_live"))
            if m.empty:
                print("   Khong co gio trung giua lich su va live de doi chieu (binh thuong neu lich su dung truoc live).")
            else:
                for c in ["price", "peg_deviation", "supply", "btc_vol"]:
                    d = (m[c + "_hist"] - m[c + "_live"]).abs()
                    print(f"   doi chieu {c}: {len(m)} dong trung, lech lon nhat {d.max():.2e}, lech trung binh {d.mean():.2e}")

    print(f"\nTONG KET: {cnt['ok']} muc dat, {cnt['loi']} muc loi.")
    print("Xong. Dan toan bo output cho Claude.")


if __name__ == "__main__":
    main()
