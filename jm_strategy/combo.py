"""四个指标的组合测试：任选 1~4 个指标（共 15 种组合），选中指标全部同向时开仓。

各指标的多空状态：
  1 SAR       快 SAR(10,2,20) 多头 / 空头
  2 BBI       收盘 > BBI / 收盘 < BBI
  3 STM       ATR 标准化 STM > 0 / < 0（含顶部/底部预警当根不追）
  4 STM简化版  未标准化 STM > 0 / < 0
离场：
  any      任一选中指标不再同向即离场
  all      全部选中指标反向才离场（并反手）
  chand    ATR 吊灯止损（持仓最高/最低价 ∓ trail_k×ATR），不依赖任何指标，所有组合通用
  sarslow  慢 SAR(10,0.5,5) 翻转离场，属于指标 1 的用法，只用于含指标 1 的组合
"""
from itertools import combinations

import pandas as pd

from .indicators import LLV, MA, REF, SAR, ind_bbi, ind_stm_v1, ind_stm_v2

NAMES = {1: "SAR", 2: "BBI", 3: "STM", 4: "STM简化"}
ALL_COMBOS = [c for k in range(1, 5) for c in combinations((1, 2, 3, 4), k)]


def indicator_states(df):
    """返回每个指标的方向状态（+1 多 / -1 空 / 0 无效）以及 ATR、预警。"""
    c = df["close"]
    _, sar_trend = SAR(df["high"], df["low"], 10, 0.02, 0.2)
    _, slow_trend = SAR(df["high"], df["low"], 10, 0.005, 0.05)
    bbi = ind_bbi(c)
    st = ind_stm_v1(df)
    stm2 = ind_stm_v2(df)

    def sign(cond_up, cond_dn):
        return cond_up.astype(int) - cond_dn.astype(int)

    states = pd.DataFrame({
        1: sar_trend,
        2: sign(c > bbi, c < bbi),
        3: sign(st["stm"] > 0, st["stm"] < 0),
        4: sign(stm2 > 0, stm2 < 0),
    }, index=df.index)
    bottom_warn = (c < REF(LLV(df["low"], 10))) & (st["stm"] > REF(st["stm"])) & (st["stm"] < 0)
    return states, st["atr"], st["top_warn"], bottom_warn, slow_trend


def combo_signals(df, combo, exit_rule="any", trend_n=0, cache=None):
    states, atr, top_warn, bottom_warn, slow_trend = cache or indicator_states(df)
    sub = states[list(combo)]
    bull = (sub == 1).all(axis=1)
    bear = (sub == -1).all(axis=1)
    if trend_n > 0:
        ma = MA(df["close"], trend_n)
        bull &= (df["close"] > ma) & (ma > REF(ma))
        bear &= (df["close"] < ma) & (ma < REF(ma))
    long_sig = bull & ~bull.shift(1, fill_value=False)
    short_sig = bear & ~bear.shift(1, fill_value=False)
    if 3 in combo:
        long_sig &= ~top_warn
        short_sig &= ~bottom_warn
    if exit_rule == "any":
        exit_long = ~(sub == 1).all(axis=1)
        exit_short = ~(sub == -1).all(axis=1)
    elif exit_rule == "all":
        exit_long, exit_short = (sub == -1).all(axis=1), (sub == 1).all(axis=1)
    elif exit_rule == "sarslow":
        exit_long, exit_short = slow_trend == -1, slow_trend == 1
    else:  # chand：只靠回测里的吊灯止损离场
        exit_long = exit_short = pd.Series(False, index=df.index)
    return pd.DataFrame({
        "atr": atr, "exit_trend": 0, "stm": 0.0, "bull": bull, "bear": bear,
        "long_sig": long_sig, "short_sig": short_sig, "exit_long": exit_long, "exit_short": exit_short,
    }, index=df.index)


def combo_name(combo):
    return "+".join(str(i) for i in combo) + " (" + "+".join(NAMES[i] for i in combo) + ")"
