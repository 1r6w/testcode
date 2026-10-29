"""SAR + BBI + STM 三重共振趋势策略（焦煤加权，日线）。

入场（收盘出信号，次日开盘成交）：
  多：STM>stm_min（指标三动能为正） 且 收盘>BBI（指标二） 且 SAR 处于多头（指标一），
      三者由“不全满足”变为“全满足”的那根 K 线开仓；当根出现顶部预警则不追。
      可选长周期过滤：收盘在 MA(trend_n) 之上且该均线上行。
  空：完全对称。
出场（exit_mode）：
  sar     SAR 翻转（收盘确认，次日开盘平仓），SAR 参数可与入场独立（exit_sar_step/exit_sar_max）
  chand   吊灯止损：持仓期间最高价 - trail_k × ATR，盘中触发
  stm     STM 反向穿越零轴（收盘确认，次日开盘平仓）
  任何模式下都有初始止损：入场价 ∓ stop_k × ATR(14)，盘中触发；
  breakeven_r>0 时，浮盈达到该 R 倍后把止损移到保本价。
平仓当根若反向共振成立，则次日开盘反手。
"""
from dataclasses import dataclass

import pandas as pd

from .indicators import CROSS, LLV, MA, REF, SAR, ind_bbi, ind_stm_v1


@dataclass
class Params:
    sar_n: int = 10
    sar_step: float = 0.02
    sar_max: float = 0.20
    bbi: tuple = (3, 6, 12, 24)
    stm_n: int = 20
    stop_k: float = 1.5          # 初始止损 ATR 倍数
    exit_mode: str = "sar"       # sar / chand / stm
    exit_sar_step: float = 0.005 # 离场用慢速 SAR（让利润奔跑）；0 = 与入场 SAR 相同
    exit_sar_max: float = 0.05
    trail_k: float = 0.0         # 吊灯止损 ATR 倍数（chand 模式必填，其他模式 >0 时作为附加保护）
    breakeven_r: float = 0.0     # 浮盈达到 N 倍 R 后止损移至保本，0 = 关闭
    trend_n: int = 60            # 长周期趋势过滤均线，0 = 关闭
    use_top_filter: bool = True  # 顶部预警/底部预警时不追单
    stm_min: float = 0.0         # STM 绝对值门槛，过滤零轴附近的弱信号


def compute_signals(df, p: Params):
    s = pd.DataFrame(index=df.index)
    s["sar"], s["sar_trend"] = SAR(df["high"], df["low"], p.sar_n, p.sar_step, p.sar_max)
    if p.exit_sar_step > 0:
        _, s["exit_trend"] = SAR(df["high"], df["low"], p.sar_n, p.exit_sar_step, p.exit_sar_max or p.sar_max)
    else:
        s["exit_trend"] = s["sar_trend"]
    s["bbi"] = ind_bbi(df["close"], *p.bbi)
    s = s.join(ind_stm_v1(df, p.stm_n))
    c = df["close"]

    # 底部预警：顶部预警的镜像
    bottom_warn = (c < REF(LLV(df["low"], 10))) & (s["stm"] > REF(s["stm"])) & (s["stm"] < 0)

    bull = (s["stm"] > p.stm_min) & (c > s["bbi"]) & (s["sar_trend"] == 1)
    bear = (s["stm"] < -p.stm_min) & (c < s["bbi"]) & (s["sar_trend"] == -1)
    if p.trend_n > 0:
        ma = MA(c, p.trend_n)
        bull &= (c > ma) & (ma > REF(ma))
        bear &= (c < ma) & (ma < REF(ma))
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
