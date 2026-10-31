"""策略库：用四个指标构建的多种独立策略（多空完全对称），统一回测比较。

记号（以用户软件中的名称为准）：
  优化1 = 指标三（ATR 标准化 STM，红/黄/绿柱）     指标1 = 指标四（简化 STM，红/绿柱）
  减速：多头视角下 STM<0 且比前一天高（绿柱变黄/变短），空头视角下 STM>0 且比前一天低
  绿变红：STM 上穿零轴；红变绿：STM 下穿零轴
  SAR 参数组：std=SAR(10,2%,20%)  s1=SAR(5,3%,30%)  s2=SAR(5,4%,40%)  s3=SAR(3,5%,50%)
             mid=SAR(10,1%,10%)  slow=SAR(10,0.5%,5%)

S1 双指标连续 k 天减速 → 平仓反手（可加高点抬升、MA60 过滤）
S2 连续 k 天减速 → 初步抄底（半仓），绿变红 → 加满仓
S3 绿变红抄底 / 红变绿摸顶（优化1、指标1 或两者都过零）
S4 灵敏 SAR 翻转反手（可加 STM / MA60 过滤）
S5 先出现双指标减速，再等灵敏 SAR 翻转确认入场
S6 两段式：3 天双减速先开半仓，灵敏 SAR 翻转或绿变红加仓
S7 1+4 趋势共振，入场 SAR 改用更灵敏参数
S8 S4 扩展版（用于提高频率）：入场可放宽为“SAR 多头且指标1>0 首次同时成立”，离场可用更快的 SAR，目标约每月一笔
离场（exit）：rev = 出现反向开仓信号即平仓（并在反向过滤满足时反手）；sar:X = SAR 组 X 在入场后反向翻转；
             可用 + 组合；S7 的 slowstate = 慢 SAR 方向与持仓相反即平（同 1+4 原规则）。
止损（stop）：swing = 近 10 天最低价/最高价外加 0.2ATR；atrX = X 倍 ATR。止损距离限制在 0.5~4 ATR。
"""
import itertools
from types import SimpleNamespace

import pandas as pd

from .indicators import HHV, LLV, MA, REF, SAR, ind_stm_v1, ind_stm_v2

SAR_SETS = {"std": (10, 0.02, 0.2), "s1": (5, 0.03, 0.3), "s2": (5, 0.04, 0.4), "s3": (3, 0.05, 0.5),
            "mid": (10, 0.01, 0.1), "slow": (10, 0.005, 0.05)}


def base_series(df):
    c, h, l = df["close"], df["high"], df["low"]
    st = ind_stm_v1(df)
    sar = {k: SAR(h, l, *v)[1] for k, v in SAR_SETS.items()}
    return SimpleNamespace(df=df, c=c, h=h, l=l, s3=st["stm"], s4=ind_stm_v2(df), atr=st["atr"],
                           ma=MA(c, 60), sar=sar)


def view(b, side):
    """把多空统一成“多头视角”：空头时指标取反、高点换成低点。"""
    up = b.ma > REF(b.ma)
    return SimpleNamespace(
        side=side, c=b.c, atr=b.atr,
        s3=side * b.s3, s4=side * b.s4,
        sar={k: side * t for k, t in b.sar.items()},
        hh=(b.h > REF(b.h)) if side > 0 else (b.l < REF(b.l)),
        ma_ok=((b.c > b.ma) & up) if side > 0 else ((b.c < b.ma) & ~up),
        swing=(LLV(b.l, 10) - 0.2 * b.atr) if side > 0 else (HHV(b.h, 10) + 0.2 * b.atr),
    )


def streak(cond, n):
    return cond.astype(int).rolling(n, min_periods=n).sum() == n


def first(cond):
    return cond & ~cond.shift(1, fill_value=False)


def slowing(x):
    return (x < 0) & (x > REF(x))


def cross_up(x):
    return (x > 0) & (REF(x) <= 0)


def flip_with(t):
    return (t == 1) & (REF(t) == -1)


def flip_against(t):
    return (t == -1) & (REF(t) == 1)


def double_slow(v, k):
    return streak(slowing(v.s3), k) & streak(slowing(v.s4), k)


# ---------------- 各策略的“多头视角”开仓规则：返回 (原始信号, 过滤后开仓信号, 加仓信号或 None) ----------------

def s1(v, p):
    core = double_slow(v, p["k"])
    if p["hh"]:
        core &= streak(v.hh, p["hh"])
    raw = first(core)
    return raw, raw & (v.ma_ok if p["filt"] == "ma" else True), None


def s2(v, p):
    core = streak(slowing(v.s3), p["k"])
    if p["ind4"] == "all":
        core &= streak(slowing(v.s4), p["k"])
    raw = first(core)
    add = cross_up(v.s3 if p["add"] == "s3" else v.s4)
    return raw, raw, add


def s3(v, p):
    if p["ind"] == "3":
        raw = cross_up(v.s3)
    elif p["ind"] == "4":
        raw = cross_up(v.s4)
    else:
        raw = first((v.s3 > 0) & (v.s4 > 0))
    f = p["filt"]
    ok = v.ma_ok if f == "ma" else (v.sar[f[4:]] == 1) if f.startswith("sar:") else True
    return raw, raw & ok, None


def s4(v, p):
    raw = flip_with(v.sar[p["sar"]])
    f = p["filt"]
    ok = {"none": True, "stm3": v.s3 > 0, "stm4": v.s4 > 0, "ma": v.ma_ok}[f]
    return raw, raw & ok, None


def s5(v, p):
    recent = HHV(first(double_slow(v, p["k"])).astype(int), p["w"]) == 1
    raw = flip_with(v.sar[p["sar"]]) & recent
    return raw, raw, None


def s6(v, p):
    raw = first(double_slow(v, 3))
    add = cross_up(v.s3) if p["add"] == "s3" else flip_with(v.sar["s1"])
    return raw, raw & (v.ma_ok if p["filt"] == "ma" else True), add


def s8(v, p):
    t = v.sar[p["sar"]]
    if p["entry"] == "flip+stm4":      # 同 S4：翻转当天指标1 已同向
        raw = flip_with(t) & (v.s4 > 0)
    elif p["entry"] == "state":        # 灵敏 SAR 多头且指标1>0，两者首次同时成立
        raw = first((t == 1) & (v.s4 > 0))
    else:                              # 只看 SAR 翻转
        raw = flip_with(t)
    return raw, raw, None


def s7(v, p):
    raw = first((v.sar[p["sar"]] == 1) & (v.s4 > 0) & v.ma_ok)
    return raw, raw, None


STRATEGIES = {
    "S1": ("双指标连续k天减速→平仓反手", s1,
           dict(k=[3, 4], hh=[0, 2], filt=["none", "ma"], exit=["rev", "rev+sar:s1", "rev+sar:std", "rev+sar:slow"],
                stop=["swing", "atr2"])),
    "S2": ("连续k天减速初步抄底(半仓)+绿变红加仓", s2,
           dict(k=[4, 5, 6], ind4=["all", "off"], add=["s3", "s4"], exit=["rev", "sar:std", "sar:slow", "rev+sar:s1"],
                stop=["swing", "atr2"])),
    "S3": ("绿变红抄底/红变绿摸顶", s3,
           dict(ind=["3", "4", "both"], filt=["none", "ma", "sar:std", "sar:s1"], exit=["rev", "sar:s1", "sar:std", "sar:slow"],
                stop=["swing", "atr2"])),
    "S4": ("灵敏SAR翻转反手", s4,
           dict(sar=["std", "s1", "s2", "s3"], filt=["none", "stm3", "stm4", "ma"], exit=["rev", "sar:slow"],
                stop=["atr1.5", "atr2.5"])),
    "S5": ("双减速后等灵敏SAR翻转确认", s5,
           dict(k=[3, 5], w=[5, 10], sar=["s1", "s2", "std"], exit=["sar:same", "sar:slow"], stop=["swing", "atr2"])),
    "S6": ("两段式:3天双减速半仓+确认加仓", s6,
           dict(add=["s3", "sarflip"], filt=["none", "ma"], exit=["sar:std", "sar:slow", "rev+sar:slow"],
                stop=["swing", "atr2"])),
    "S7": ("1+4趋势共振+更灵敏的入场SAR", s7,
           dict(sar=["std", "s1", "s2", "s3"], exit=["slowstate", "sar:slow"], stop=["atr1.5", "atr2.5"])),
    "S8": ("S4扩展版(入场/离场放宽)", s8,
           dict(sar=["s1", "s2", "s3", "std"], entry=["flip+stm4", "state", "flip"],
                exit=["sar:same", "sar:std", "sar:mid", "sar:slow", "rev"], stop=["atr1.5", "atr2", "atr2.5"])),
}


def build_signals(b, code, p):
    """组装成回测引擎需要的信号表（多空两个视角各算一遍）。"""
    _, fn, _ = STRATEGIES[code]
    views = {1: view(b, 1), -1: view(b, -1)}
    res = {side: fn(views[side], p) for side in (1, -1)}
    out = pd.DataFrame(index=b.df.index)
    out["atr"], out["exit_trend"], out["stm"] = b.atr, 0, b.s3
    out["long_sig"] = out["bull"] = res[1][1].astype(bool)
    out["short_sig"] = out["bear"] = res[-1][1].astype(bool)
    for side, name in ((1, "exit_long"), (-1, "exit_short")):
        v, ex = views[side], pd.Series(False, index=b.df.index)
        for part in p["exit"].split("+"):
            if part == "rev":
                ex |= res[-side][0]
            elif part == "slowstate":
                ex |= v.sar["slow"] == -1
            elif part.startswith("sar:"):
                key = p["sar"] if part == "sar:same" else part[4:]
                ex |= flip_against(v.sar[key])
        out[name] = ex
    if p["stop"] == "swing":
        out["long_stop"], out["short_stop"] = views[1].swing, views[-1].swing
    else:
        out.attrs["stop_k"] = float(p["stop"][3:])  # ATR 止损按成交价计算，交给回测引擎
    if res[1][2] is not None:
        out["add_long"], out["add_short"] = res[1][2], res[-1][2]
        out.attrs["entry_frac"] = 0.5
    return out


def backtest_params(sig):
    from .strategy import Params
    return Params(exit_mode="signal", stop_k=sig.attrs.get("stop_k", 2.0))


def variants(code):
    grid = STRATEGIES[code][2]
    for vals in itertools.product(*grid.values()):
        yield dict(zip(grid, vals))
