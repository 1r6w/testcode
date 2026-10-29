"""动能拐点策略：在下跌/上涨“减速”时提前反手（用户思路）。

指标对应（以用户软件中的名称为准）：
  “优化的1指标” = 指标三，ATR 标准化 STM，柱色：红(>0 且升) 黄(>0 且降 / <0 且升) 绿(<0 且降)
  “1指标”       = 指标四，未标准化 STM，柱色：红(>0) 绿(<0)

抄底拐点（核心条件）：
  ① 优化的1指标 STM<0 且连续 k 天上升（绿柱变黄柱，下跌减速）
  ② 1指标 STM<0 且同期也在上升（下跌趋势也减缓），ind4="all" 要求 k 天都升，"last" 只要求当天升，"off" 不要求
  ③ 高点连续抬升：最近 hh_n 次比较中每天最高价都高于前一天（3 天高点不断抬升 = 2 次比较）
逃顶拐点：完全对称（红柱变黄、1指标红柱变短、低点连续下移）。
核心条件首次全部满足的那根 K 线收盘出信号，次日开盘成交。

开仓另加可选过滤（只影响开新仓，不影响平仓）：
  depth  最近 10 天 优化的1指标 极值超过 depth（过滤零轴附近的小波动）
  trend  with = 只在 MA60 向上时抄底、向下时逃顶（顺大势）；against = 反之
止损：swing = 最近 swing_n 天最低价（空单为最高价）外加 0.2 ATR 缓冲（距离限制 0.5~4 ATR）；atr = stop_k × ATR。
离场：
  turn        反向拐点（核心条件）出现即平仓，若反向开仓过滤也满足则同时反手
  slow        慢 SAR(10,0.005,0.05) 在入场后翻转才平仓
  turn+slow   两者任一
  chand       ATR 吊灯止损（持仓极值 ∓ trail_k×ATR）
  turn+chand  两者任一
"""
from dataclasses import dataclass

import pandas as pd

from .indicators import HHV, LLV, MA, REF, SAR, ind_stm_v1, ind_stm_v2


@dataclass
class TurnParams:
    k: int = 3               # 连续减速天数
    hh_n: int = 2            # 高点连续抬升的比较次数，0 = 不要求
    ind4: str = "all"        # all / last / off
    depth: float = 0.0       # 开仓过滤：优化的1指标 近 10 天极值的最小深度
    trend: str = "none"      # 开仓过滤：none / with / against
    stop_mode: str = "swing" # swing / atr
    swing_n: int = 10
    stop_k: float = 2.0
    exit: str = "turn"       # turn / slow / turn+slow / chand / turn+chand
    trail_k: float = 4.0
    breakeven_r: float = 0.0 # 浮盈达到 N 倍风险后止损移到保本


def _streak(cond, n):
    return cond.astype(int).rolling(n, min_periods=n).sum() == n


def _first(cond):
    return cond & ~cond.shift(1, fill_value=False)


def turn_signals(df, tp: TurnParams):
    c, h, l = df["close"], df["high"], df["low"]
    st = ind_stm_v1(df)
    s3, atr = st["stm"], st["atr"]
    s4 = ind_stm_v2(df)

    up3 = (s3 < 0) & (s3 > REF(s3))       # 优化的1指标：绿变黄（下跌减速）
    dn3 = (s3 > 0) & (s3 < REF(s3))       # 优化的1指标：红变黄（上涨减速）
    up4 = (s4 < 0) & (s4 > REF(s4))       # 1指标：绿柱变短
    dn4 = (s4 > 0) & (s4 < REF(s4))       # 1指标：红柱变短

    long_core = _streak(up3, tp.k)
    short_core = _streak(dn3, tp.k)
    if tp.ind4 == "all":
        long_core &= _streak(up4, tp.k)
        short_core &= _streak(dn4, tp.k)
    elif tp.ind4 == "last":
        long_core &= up4
        short_core &= dn4
    if tp.hh_n > 0:
        long_core &= _streak(h > REF(h), tp.hh_n)
        short_core &= _streak(l < REF(l), tp.hh_n)
    long_turn, short_turn = _first(long_core), _first(short_core)

    long_ok = pd.Series(True, index=df.index)
    short_ok = pd.Series(True, index=df.index)
    if tp.depth > 0:
        long_ok &= LLV(s3, 10) < -tp.depth
        short_ok &= HHV(s3, 10) > tp.depth
    if tp.trend != "none":
        ma = MA(c, 60)
        rising = ma > REF(ma)
        if tp.trend == "with":
            long_ok &= rising
            short_ok &= ~rising
        else:
            long_ok &= ~rising
            short_ok &= rising
    long_sig, short_sig = long_turn & long_ok, short_turn & short_ok

    none = pd.Series(False, index=df.index)
    exit_long = short_turn if "turn" in tp.exit else none
    exit_short = long_turn if "turn" in tp.exit else none
    if "slow" in tp.exit:
        _, tr = SAR(h, l, 10, 0.005, 0.05)
        exit_long = exit_long | ((tr == -1) & (REF(tr) == 1))   # 入场后出现的翻转
        exit_short = exit_short | ((tr == 1) & (REF(tr) == -1))

    out = pd.DataFrame({
        "atr": atr, "exit_trend": 0, "stm": s3, "bull": long_sig, "bear": short_sig,
        "long_sig": long_sig, "short_sig": short_sig, "exit_long": exit_long, "exit_short": exit_short,
    }, index=df.index)
    if tp.stop_mode == "swing":
        out["long_stop"] = LLV(l, tp.swing_n) - 0.2 * atr
        out["short_stop"] = HHV(h, tp.swing_n) + 0.2 * atr
    return out


def turn_backtest_params(tp: TurnParams):
    """把 TurnParams 转成回测引擎的 Params。"""
    from .strategy import Params
    return Params(exit_mode="signal", stop_k=tp.stop_k, breakeven_r=tp.breakeven_r,
                  trail_k=tp.trail_k if "chand" in tp.exit else 0.0)
