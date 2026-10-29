"""混合策略：1+4 趋势共振（主信号） + 用户的动能拐点（提前平仓 / 平仓抄底 / 平仓开空）。

趋势单（第一套信号）：快 SAR + STM简化 同向 + MA60 过滤开仓，慢 SAR 反向离场，止损 2.5 ATR（即 1+4 最优组合）。
拐点信号见 turn.py（优化的1指标连续 k 天减速 + 1指标同步减速 + 高点抬升 / 低点下移）。
mode：
  exit   H1 拐点只用于提前平掉趋势单（持多遇逃顶拐点平多，持空遇抄底拐点平空）
  entry  H2 拐点只用于开新仓（空仓时）
  both   H3 两者都用：持空遇抄底拐点 → 平空开多；持多遇逃顶拐点 → 平多开空
拐点单（第二套信号）的离场：反向拐点 / 入场后慢 SAR 翻转 / 反向趋势共振，止损为前低前高或 ATR。
"""
from dataclasses import dataclass

import pandas as pd

from .combo import combo_signals
from .indicators import REF, SAR
from .turn import TurnParams, turn_signals


@dataclass
class HybridParams:
    mode: str = "both"           # exit / entry / both
    k: int = 3
    hh_n: int = 2
    ind4: str = "all"
    turn_trend: str = "none"     # 拐点开仓过滤：none / with（只在 MA60 同向时开拐点单）
    turn_stop: str = "swing"     # swing / atr
    turn_stop_k: float = 2.0
    turn_exit: str = "turn+slow" # turn / slow / turn+slow
    trend_stop_k: float = 2.5


def hybrid_signals(df, hp: HybridParams, cache=None):
    trend = combo_signals(df, (1, 4), "sarslow", 60, cache)
    tp = TurnParams(k=hp.k, hh_n=hp.hh_n, ind4=hp.ind4, trend=hp.turn_trend,
                    stop_mode=hp.turn_stop, stop_k=hp.turn_stop_k, exit="turn")
    turn = turn_signals(df, tp)
    raw_long_turn, raw_short_turn = turn["exit_short"], turn["exit_long"]  # 未过滤的拐点

    sig = trend.copy()
    if hp.mode in ("exit", "both"):
        sig["exit_long"] = trend["exit_long"] | raw_short_turn
        sig["exit_short"] = trend["exit_short"] | raw_long_turn
    none = pd.Series(False, index=df.index)
    if hp.mode in ("entry", "both"):
        sig["long_sig2"], sig["short_sig2"] = turn["long_sig"], turn["short_sig"]
    else:
        sig["long_sig2"] = sig["short_sig2"] = none

    ex_l = trend["short_sig"].copy()   # 反向趋势共振
    ex_s = trend["long_sig"].copy()
    if "turn" in hp.turn_exit:
        ex_l |= raw_short_turn
        ex_s |= raw_long_turn
    if "slow" in hp.turn_exit:
        _, tr = SAR(df["high"], df["low"], 10, 0.005, 0.05)
        ex_l |= (tr == -1) & (REF(tr) == 1)
        ex_s |= (tr == 1) & (REF(tr) == -1)
    sig["exit_long2"], sig["exit_short2"] = ex_l, ex_s
    if hp.turn_stop == "swing":
        sig["long_stop2"], sig["short_stop2"] = turn["long_stop"], turn["short_stop"]
    else:
        sig["long_stop2"] = df["close"] - hp.turn_stop_k * trend["atr"]
        sig["short_stop2"] = df["close"] + hp.turn_stop_k * trend["atr"]
    return sig


def hybrid_backtest_params(hp: HybridParams):
    from .strategy import Params
    return Params(exit_mode="signal", stop_k=hp.trend_stop_k)
