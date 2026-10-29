"""逐 K 线事件驱动回测：收盘出信号、次日开盘成交、止损盘中触发，含手续费与滑点。"""
from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from .strategy import Params, compute_signals


@dataclass
class Account:
    capital: float = 1_000_000
    mult: int = 60               # 焦煤合约乘数 60 吨/手
    tick: float = 0.5            # 最小变动价位
    slip_ticks: int = 1          # 每边滑点
    fee_rate: float = 1.5e-4     # 每边手续费（按成交额，含交易所+期货公司）
    risk_per_trade: float = 0.02 # 每笔初始止损风险占权益比例
    max_leverage: float = 3.0    # 名义市值 / 权益 上限（焦煤保证金约 15%-20%）


def run_backtest(df, p: Params = None, acct: Account = None, start=None, end=None, sig=None):
    """sig 为空时按 Params 计算信号；也可传入外部信号（需含 atr/exit_trend/stm/bull/bear/long_sig/short_sig，
    exit_mode="signal" 时另需 exit_long/exit_short）。"""
    p, acct = p or Params(), acct or Account()
    sig = compute_signals(df, p) if sig is None else sig
    if start or end:
        df, sig = df.loc[start:end], sig.loc[start:end]
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = sig["atr"].to_numpy(float)
    trend = sig["exit_trend"].to_numpy(int)
    stm = sig["stm"].to_numpy(float)
    bull, bear = sig["bull"].to_numpy(bool), sig["bear"].to_numpy(bool)
    if p.exit_mode == "signal":
        exit_long, exit_short = sig["exit_long"].to_numpy(bool), sig["exit_short"].to_numpy(bool)
    # 可选：信号自带止损价位（如前低/前高），否则用 stop_k × ATR
    long_stop = sig["long_stop"].to_numpy(float) if "long_stop" in sig else None
    short_stop = sig["short_stop"].to_numpy(float) if "short_stop" in sig else None
    long_sig, short_sig = sig["long_sig"].to_numpy(bool), sig["short_sig"].to_numpy(bool)
    dates = df.index
    slip = acct.slip_ticks * acct.tick

    cash = acct.capital
    pos = 0              # 正多负空，单位：手
    entry_px = stop = ext = 0.0
    entry_i = 0
    pending_exit, exit_reason = False, ""
    pending_entry = 0    # +1 / -1
    pending_atr = 0.0
    pending_level = np.nan  # 信号自带的止损价位
    trades, equity = [], np.empty(len(df))

    def fee(px, lots):
        return px * lots * acct.mult * acct.fee_rate

    def close_pos(i, px, reason):
        nonlocal cash, pos
        side = 1 if pos > 0 else -1
        fill = px - side * slip
        pnl = (fill - entry_px) * pos * acct.mult - fee(fill, abs(pos))
        cash += pnl
        trades.append(dict(entry_date=dates[entry_i], exit_date=dates[i], side="多" if side > 0 else "空",
                           lots=abs(pos), entry=entry_px, exit=fill, pnl=pnl,
                           R=(fill - entry_px) * side / init_risk, bars=i - entry_i, reason=reason))
        pos = 0

    init_risk = 1.0
    for i in range(len(df)):
        # ---- 开盘：执行挂单 ----
        if pending_exit and pos != 0:
            close_pos(i, o[i], exit_reason)
        pending_exit = False
        if pending_entry and pos == 0 and pending_atr > 0:
            side = pending_entry
            fill = o[i] + side * slip
            dist = p.stop_k * pending_atr
            if not np.isnan(pending_level):  # 前低/前高止损，距离限制在 0.5~4 倍 ATR
                dist = min(max((fill - pending_level) * side, 0.5 * pending_atr), 4.0 * pending_atr)
            equity_now = cash
            lots = math.floor(equity_now * acct.risk_per_trade / (dist * acct.mult))
            lots = min(lots, math.floor(equity_now * acct.max_leverage / (fill * acct.mult)))
            if lots > 0:
                pos, entry_px, entry_i, init_risk = side * lots, fill, i, dist
                stop, ext = fill - side * dist, fill
                cash -= fee(fill, lots)
        pending_entry = 0

        # ---- 盘中：止损（初始/保本/吊灯） ----
        if pos > 0 and l[i] <= stop:
            close_pos(i, min(o[i], stop), "止损" if stop < entry_px else "跟踪止盈")
        elif pos < 0 and h[i] >= stop:
            close_pos(i, max(o[i], stop), "止损" if stop > entry_px else "跟踪止盈")

        # ---- 收盘：更新跟踪止损、判断离场、生成信号 ----
        if pos != 0:
            side = 1 if pos > 0 else -1
            ext = max(ext, h[i]) if side > 0 else min(ext, l[i])
            if p.trail_k > 0:
                trail = ext - side * p.trail_k * atr[i]
                stop = max(stop, trail) if side > 0 else min(stop, trail)
            if p.breakeven_r > 0 and (ext - entry_px) * side >= p.breakeven_r * init_risk:
                stop = max(stop, entry_px) if side > 0 else min(stop, entry_px)
            if p.exit_mode == "sar" and trend[i] == -side:
                pending_exit, exit_reason = True, "SAR翻转"
            elif p.exit_mode == "stm" and stm[i] * side < 0:
                pending_exit, exit_reason = True, "STM穿零"
            elif p.exit_mode == "signal" and (exit_long[i] if side > 0 else exit_short[i]):
                pending_exit, exit_reason = True, "指标离场"

        flat_next = pos == 0 or pending_exit
        if flat_next and i + 1 < len(df):
            if long_sig[i] or (pending_exit and pos < 0 and bull[i]):
                pending_entry, pending_atr = 1, atr[i]
                pending_level = long_stop[i] if long_stop is not None else np.nan
            elif short_sig[i] or (pending_exit and pos > 0 and bear[i]):
                pending_entry, pending_atr = -1, atr[i]
                pending_level = short_stop[i] if short_stop is not None else np.nan

        equity[i] = cash + ((c[i] - entry_px) * pos * acct.mult if pos else 0.0)

    if pos != 0:
        close_pos(len(df) - 1, c[-1], "回测结束")
        equity[-1] = cash

    eq = pd.Series(equity, index=dates, name="equity")
    tr = pd.DataFrame(trades)
    return eq, tr, sig


def metrics(eq, tr, capital):
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    final = eq.iloc[-1]
    cagr = (final / capital) ** (1 / years) - 1 if final > 0 else -1.0
    ret = eq.pct_change().dropna()
    dd = eq / eq.cummax() - 1
    m = {
        "起止": f"{eq.index[0].date()} ~ {eq.index[-1].date()}",
        "年数": round(years, 2),
        "期末权益": round(final, 0),
        "总收益率": final / capital - 1,
        "年化收益率": cagr,
        "最大回撤": dd.min(),
        "卡玛比率": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "夏普比率": ret.mean() / ret.std() * np.sqrt(244) if ret.std() > 0 else np.nan,
        "交易次数": len(tr),
    }
    if len(tr):
        win, loss = tr[tr.pnl > 0], tr[tr.pnl <= 0]
        avg_w = win.pnl.mean() if len(win) else 0.0
        avg_l = -loss.pnl.mean() if len(loss) else np.nan
        m.update({
            "胜率": len(win) / len(tr),
            "盈亏比(均盈/均亏)": avg_w / avg_l if avg_l and avg_l > 0 else np.nan,
            "盈亏比(按R)": win.R.mean() / -loss.R.mean() if len(win) and len(loss) and loss.R.mean() < 0 else np.nan,
            "平均R(均盈)": win.R.mean() if len(win) else 0.0,
            "平均R(均亏)": loss.R.mean() if len(loss) else 0.0,
            "利润因子": win.pnl.sum() / -loss.pnl.sum() if loss.pnl.sum() < 0 else np.nan,
            "期望值R/笔": tr.R.mean(),
            "最大连亏次数": _max_streak(tr.pnl <= 0),
            "平均持仓K线": tr.bars.mean(),
            "年均交易次数": len(tr) / years,
        })
    return m


def _max_streak(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def yearly_returns(eq):
    y = eq.resample("YE").last()
    first = eq.iloc[0]
    prev = pd.concat([pd.Series([first], index=[eq.index[0]]), y.iloc[:-1]])
    return pd.Series(y.to_numpy() / prev.to_numpy() - 1, index=y.index.year, name="年度收益")
