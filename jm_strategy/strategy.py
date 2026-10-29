"""SAR + BBI + STM 三重共振趋势策略（焦煤加权，日线）。

入场（收盘出信号，次日开盘成交）：
  多：STM>0（指标三动能为正） 且 收盘>BBI（指标二） 且 SAR 处于多头（指标一），
      三者由“不全满足”变为“全满足”的那根 K 线开仓；当根出现顶部预警则不追。
  空：完全对称。
出场：
  1. 初始止损：入场价 ∓ stop_k × ATR(14)，盘中触发（跳空则按开盘价成交）。
  2. 移动止盈：SAR 翻转（收盘确认，次日开盘平仓）；可选 chandelier 吊灯止损 trail_k × ATR。
  3. 平仓当根若反向共振成立，则次日开盘反手。
小亏大赚：止损固定、盈利仓位交给 SAR 一路跟踪，从而拉高盈亏比。
"""
from dataclasses import dataclass

import pandas as pd

from .indicators import SAR, ind_bbi, ind_stm_v1, CROSS


@dataclass
class Params:
    sar_n: int = 10
    sar_step: float = 0.02
    sar_max: float = 0.20
    bbi: tuple = (3, 6, 12, 24)
    stm_n: int = 20
    stop_k: float = 2.0          # 初始止损 ATR 倍数
    trail_k: float = 0.0         # 吊灯止损 ATR 倍数，0 = 关闭，仅用 SAR 跟踪
    use_top_filter: bool = True  # 顶部预警/底部预警时不追单
    stm_min: float = 0.0         # STM 绝对值门槛，过滤零轴附近的弱信号


def compute_signals(df, p: Params):
    s = pd.DataFrame(index=df.index)
    s["sar"], s["sar_trend"] = SAR(df["high"], df["low"], p.sar_n, p.sar_step, p.sar_max)
    s["bbi"] = ind_bbi(df["close"], *p.bbi)
    st = ind_stm_v1(df, p.stm_n)
    s = s.join(st)
    c = df["close"]

    # 底部预警：顶部预警的镜像
    from .indicators import REF, LLV
    bottom_warn = (c < REF(LLV(df["low"], 10))) & (s["stm"] > REF(s["stm"])) & (s["stm"] < 0)

    bull = (s["stm"] > p.stm_min) & (c > s["bbi"]) & (s["sar_trend"] == 1)
    bear = (s["stm"] < -p.stm_min) & (c < s["bbi"]) & (s["sar_trend"] == -1)
    long_sig = bull & ~bull.shift(1, fill_value=False)
    short_sig = bear & ~bear.shift(1, fill_value=False)
    if p.use_top_filter:
        long_sig &= ~s["top_warn"]
        short_sig &= ~bottom_warn
    s["bull"], s["bear"] = bull, bear
    s["long_sig"], s["short_sig"] = long_sig, short_sig
    s["stm_cross_up"] = CROSS(s["stm"], 0)
    s["stm_cross_dn"] = CROSS(0, s["stm"])
    return s
