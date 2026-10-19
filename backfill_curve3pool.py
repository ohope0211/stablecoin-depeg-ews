#!/usr/bin/env python3
"""
backfill_curve3pool.py - BUOC 5b (thay cho Dune): lay so du DAI/USDC/USDT trong Curve 3pool theo thoi gian.

Cach lam: voi moi moc thoi gian, tim block cuoi cung cua gio do roi goi ham balances(i) cua pool 3pool
tai block do (can node ARCHIVE - Alchemy ban free co ho tro). Day cung la ham se dung o pipeline live (Buoc 6).

Chuan bi:
  1. Dang ky tai khoan mien phi o alchemy.com, tao app mang "Ethereum Mainnet", copy HTTPS URL
     (dang https://eth-mainnet.g.alchemy.com/v2/XXXX).
  2. pip install requests

Chay (Windows PowerShell):
  $env:ETH_RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/XXXX"
  python backfill_curve3pool.py --step-hours 6
Co the dung bat cu luc nao (Ctrl+C), chay lai se TIEP TUC tu cho dang do.

Dau ra: data/raw/curve3pool_hourly.csv (cot: hour, *_balance, share_*)
  hour = gio BAT DAU nen (giong timestamp cua panel); so du la gia tri CUOI gio do.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

POOL = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"
SELECTOR = "0x4903b0d1"  # balances(uint256)
DECIMALS = [18, 6, 6]  # index 0=DAI, 1=USDC, 2=USDT
OUT = os.path.join("data", "raw", "curve3pool_hourly.csv")
FIELDS = ["hour", "dai_balance", "usdc_balance", "usdt_balance", "share_dai", "share_usdc", "share_usdt"]
S = requests.Session()
ARCHIVE_HINTS = ("missing trie", "archive", "pruned", "historical state", "state is not available")


def rpc(url, calls):
    """Gui 1 lo (batch) JSON-RPC. calls = [(method, params), ...]. Tra ve ket qua theo dung thu tu."""
    payload = [{"jsonrpc": "2.0", "id": i, "method": m, "params": p} for i, (m, p) in enumerate(calls)]
    for attempt in range(6):
        try:
            r = S.post(url, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):  # loi chung (vd sai API key)
                raise RuntimeError(str(data)[:200])
            data.sort(key=lambda x: x["id"])
            for x in data:
                if "error" in x:
                    msg = str(x["error"])
                    if any(h in msg.lower() for h in ARCHIVE_HINTS):
                        sys.exit("Node nay KHONG phai archive nen khong doc duoc so du cu. "
                                 "Hay dung Alchemy (goi free co archive). Chi tiet: " + msg[:150])
                    raise RuntimeError(msg[:200])
            return [x["result"] for x in data]
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"   loi ({str(e)[:100]}), thu lai sau {wait}s")
            time.sleep(wait)
    sys.exit("RPC that bai nhieu lan, kiem tra URL/API key va mang.")


def block_times(url, nums):
    res = rpc(url, [("eth_getBlockByNumber", [hex(n), False]) for n in nums])
    return [int(b["timestamp"], 16) for b in res]


def find_block(url, target, guess, latest):
    """Block lon nhat co timestamp <= target. Dung uoc luong roi hieu chinh dan."""
    b = max(1, min(int(guess), latest - 1))
    for it in range(40):
        t0, t1 = block_times(url, [b, b + 1])
        slow = it >= 10  # neu khong hoi tu thi nhich tung block
        if t0 > target:
            b -= 1 if slow else max(1, (t0 - target) // 12)
        elif t1 <= target:
            b += 1 if slow else max(1, (target - t1) // 12 + 1)
            if b >= latest:
                return latest
        else:
            return b
    sys.exit("Khong tim duoc block, thu chay lai.")


def read_pool(url, block):
    calls = [("eth_call", [{"to": POOL, "data": SELECTOR + f"{i:064x}"}, hex(block)]) for i in range(3)]
    res = rpc(url, calls)
    return [int(x, 16) / 10 ** d for x, d in zip(res, DECIMALS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", default=os.environ.get("ETH_RPC_URL"), help="URL RPC archive (hoac dat ETH_RPC_URL)")
    ap.add_argument("--start", default="2021-06-01", help="gio bat dau (UTC), lay som de tinh trung binh truot")
    ap.add_argument("--step-hours", type=int, default=6, help="khoang cach giua 2 lan lay (gio). 1 = day du nhat")
    args = ap.parse_args()
    if not args.rpc:
        sys.exit("Chua co URL RPC. Dat bien ETH_RPC_URL hoac dung --rpc URL.")

    t = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    step = timedelta(hours=args.step_hours)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):  # tiep tuc
        with open(OUT, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            t = datetime.strptime(rows[-1]["hour"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) + step
            print(f"Tiep tuc tu {t:%Y-%m-%d %H:%M} (da co {len(rows)} dong)")
    else:
        with open(OUT, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    latest = int(rpc(args.rpc, [("eth_blockNumber", [])])[0], 16)
    latest_ts = block_times(args.rpc, [latest])[0]
    total = max(0, int((datetime.fromtimestamp(latest_ts, timezone.utc) - t) / step))
    print(f"Block moi nhat {latest}. Con khoang {total} moc can lay (moi {args.step_hours} gio).")

    prev_b, prev_ts, rate = None, None, None
    n, t_start = 0, time.time()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        while True:
            target = int((t + timedelta(hours=1)).timestamp()) - 1  # giay cuoi cung cua gio t
            if target > latest_ts:
                break
            if prev_b is None:
                guess = latest - (latest_ts - target) / 12.3
            else:
                guess = prev_b + (target - prev_ts) / (rate or 12.5)
            b = find_block(args.rpc, target, guess, latest)
            if prev_b is not None and b > prev_b:
                rate = (target - prev_ts) / (b - prev_b)
            prev_b, prev_ts = b, target
            dai, usdc, usdt = read_pool(args.rpc, b)
            tot = dai + usdc + usdt
            w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"), dai, usdc, usdt, dai / tot, usdc / tot, usdt / tot])
            f.flush()
            n += 1
            if n % 50 == 0:
                el = time.time() - t_start
                print(f"  {t:%Y-%m-%d %H:%M} | da lay {n}/{total} | con khoang {(total - n) * el / n / 60:.0f} phut")
            t += step
    print(f"Xong: {OUT}")
    print("Dan 5 dong dau, 5 dong cuoi va cac dong quanh 2023-03-11 cho Claude.")


if __name__ == "__main__":
    main()
