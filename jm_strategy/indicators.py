"""通达信函数与四个原始指标的 Python 实现（与通达信公式口径保持一致）。"""
import numpy as np
import pandas as pd


# ---------- 通达信基础函数 ----------
def MA(s, n):
    return s.rolling(n, min_periods=n).mean()


def EMA(s, n):
    return s.ewm(span=n, adjust=False).mean()


def SMA(s, n, m):
    """通达信 SMA(X,N,M): Y = (M*X + (N-M)*Y') / N"""
    return s.ewm(alpha=m / n, adjust=False).mean()


def REF(s, n=1):
    return s.shift(n)


def HHV(s, n):
    return s.rolling(n, min_periods=1).max()


def LLV(s, n):
    return s.rolling(n, min_periods=1).min()


def CROSS(a, b):
    if not isinstance(a, pd.Series):
        a = pd.Series(a, index=b.index)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def SAR(high, low, n=10, step=0.02, maxaf=0.2):
    """抛物线转向 SAR(N,S,M)。返回 (sar, trend)，trend=1 多头(SAR在价格下方)，-1 空头。

    sar[t] 只用到 t-1 及以前的数据确定；若当根 K 线击穿 sar[t] 则在 t 收盘后翻转，
    翻转后的新值记在 t 上（与通达信画法一致）。
    """
    h, l = high.to_numpy(float), low.to_numpy(float)
    size = len(h)
    sar = np.full(size, np.nan)
    trend = np.zeros(size, dtype=int)
    if size <= n:
        return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)

    up = (h[n] + l[n]) >= (h[0] + l[0])
    if up:
        cur, ep = l[: n + 1].min(), h[: n + 1].max()
    else:
        cur, ep = h[: n + 1].max(), l[: n + 1].min()
    af = step
    sar[n], trend[n] = cur, 1 if up else -1

    for t in range(n + 1, size):
        nxt = cur + af * (ep - cur)
        if up:
            nxt = min(nxt, l[t - 1], l[t - 2])
            if l[t] < nxt:                       # 跌破 -> 翻空
                up, nxt, ep, af = False, max(ep, h[t]), l[t], step
            elif h[t] > ep:
                ep, af = h[t], min(af + step, maxaf)
        else:
            nxt = max(nxt, h[t - 1], h[t - 2])
            if h[t] > nxt:                       # 升破 -> 翻多
                up, nxt, ep, af = True, min(ep, l[t]), h[t], step
            elif l[t] < ep:
                ep, af = l[t], min(af + step, maxaf)
        cur = nxt
        sar[t], trend[t] = cur, 1 if up else -1
    return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)


# ---------- 四个原始指标 ----------
def ind_bbi(close, m1=3, m2=6, m3=12, m4=24):
    """指标二：BBI 多空指标"""
    return (MA(close, m1) + MA(close, m2) + MA(close, m3) + MA(close, m4)) / 4


def ind_stm_v1(df, n=20):
    """指标三：ATR 标准化综合动能 STM（含动能衰减、顶部预警）"""
    c, h, l = df["close"], df["high"], df["low"]
    tr = pd.concat([h - l, (h - REF(c)).abs(), (l - REF(c)).abs()], axis=1).max(axis=1)
    a = MA(tr, 14).clip(lower=0.01)
    t1 = (MA(c, 5) - MA(c, n)) / a * 20
    dif = EMA(c, 12) - EMA(c, 26)
    dea = EMA(dif, 9)
    m1 = (dif - dea) / a * 10
    diff = c - REF(c)
    rsi = SMA(diff.clip(lower=0), 14, 1) / SMA(diff.abs(), 14, 1).clip(lower=0.01) * 100
    stm = EMA(t1 + m1 + (rsi - 50) * 0.5, 3)
    out = pd.DataFrame(index=df.index)
    out["stm"] = stm
    out["atr"] = a
    out["weak"] = (stm > 0) & (stm < REF(stm)) & (REF(stm) < REF(stm, 2))       # 动能减弱
    out["top_warn"] = (c > REF(HHV(h, 10))) & (stm < REF(stm)) & (stm > 0)       # 顶部预警
    return out


def ind_stm_v2(df, n=20):
    """指标四：未标准化的简化版 STM（价格量纲，跨品种/跨年代不可比，策略中仅作参考）"""
    c = df["close"]
    dif = EMA(c, 12) - EMA(c, 26)
    dea = EMA(dif, 9)
    diff = c - REF(c)
    rsi = SMA(diff.clip(lower=0), 14, 1) / SMA(diff.abs(), 14, 1) * 100
    return (MA(c, 5) - MA(c, n)) + (dif - dea) * 5 + (rsi - 50) / 2
