"""焦煤（大商所 JM）加权指数数据：下载单合约日线并按持仓量加权合成。

加权指数（通达信 JML9 / 文华 JM888 口径）：每日所有在交易合约的 OHLC 按当日持仓量加权平均，
成交量、持仓量为各合约之和。加权指数无换月跳空，适合做信号回测。
"""
import os
import time

import pandas as pd

COLS = ["open", "high", "low", "close", "volume", "hold"]


def load_csv(path):
    """读取本地 CSV。需包含 date/open/high/low/close 列（中文列名 日期/开盘/最高/最低/收盘 也可）。"""
    df = pd.read_csv(path)
    rename = {"日期": "date", "时间": "date", "开盘": "open", "开盘价": "open", "最高": "high",
              "最高价": "high", "最低": "low", "最低价": "low", "收盘": "close", "收盘价": "close",
              "成交量": "volume", "持仓量": "hold", "持仓": "hold"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _contract_codes(start_year=2013, end_year=None):
    end_year = end_year or pd.Timestamp.today().year + 1
    return [f"JM{str(y)[2:]}{m:02d}" for y in range(start_year, end_year + 1) for m in range(1, 13)]


def fetch_contracts(cache_dir, start_year=2013, end_year=None, pause=0.3):
    """用 akshare（新浪源）逐个下载 JM 单合约日线，缓存到 cache_dir。"""
    import akshare as ak

    os.makedirs(cache_dir, exist_ok=True)
    frames = {}
    for code in _contract_codes(start_year, end_year):
        fp = os.path.join(cache_dir, f"{code}.csv")
        if os.path.exists(fp):
            d = pd.read_csv(fp, parse_dates=["date"])
        else:
            try:
                d = ak.futures_zh_daily_sina(symbol=code)
            except Exception as e:  # 未上市/不存在的合约
                print(f"skip {code}: {type(e).__name__}")
                continue
            if d is None or d.empty:
                continue
            d.to_csv(fp, index=False)
            time.sleep(pause)
            d["date"] = pd.to_datetime(d["date"])
        frames[code] = d
    return frames


def build_weighted_index(frames):
    """按持仓量加权合成指数。frames: {合约代码: DataFrame(date,open,high,low,close,volume,hold)}"""
    parts = []
    for code, d in frames.items():
        d = d.copy()
        d["date"] = pd.to_datetime(d["date"])
        for c in COLS:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["code"] = code
        parts.append(d[["date", "code"] + COLS])
    allc = pd.concat(parts, ignore_index=True)
    allc = allc[(allc["hold"] > 0) & (allc["close"] > 0)]

    def agg(g):
        w = g["hold"] / g["hold"].sum()
        return pd.Series({
            "open": (g["open"] * w).sum(), "high": (g["high"] * w).sum(),
            "low": (g["low"] * w).sum(), "close": (g["close"] * w).sum(),
            "volume": g["volume"].sum(), "hold": g["hold"].sum(),
        })

    idx = allc.groupby("date")[COLS].apply(agg).sort_index()
    # 加权后 high/low 需重新包住 open/close
    idx["high"] = idx[["high", "open", "close"]].max(axis=1)
    idx["low"] = idx[["low", "open", "close"]].min(axis=1)
    return idx


def get_jm_weighted(data_dir="data", refresh=False):
    out = os.path.join(data_dir, "JM_weighted.csv")
    if os.path.exists(out) and not refresh:
        return load_csv(out)
    frames = fetch_contracts(os.path.join(data_dir, "contracts"))
    if not frames:
        raise RuntimeError("未下载到任何合约数据：请检查网络是否能访问 stock2.finance.sina.com.cn，"
                           "或用 --csv 指定从通达信/文华导出的 JM 加权日线。")
    idx = build_weighted_index(frames)
    idx.reset_index().to_csv(out, index=False)
    return idx
