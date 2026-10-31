"""S4 策略（灵敏 SAR(5,3,30) 翻转 + 指标1 确认，2.5ATR 止损，慢 SAR 翻转离场）按指定资金与仓位方式回测，
输出交易核心数据、年度收益、逐笔交易和资金曲线。

用法：
  python run_s4.py --csv data/JM_weighted_user.csv --capital 500000 --sizing margin --frac 0.5 --margin-rate 0.15
  sizing: margin = 每次开仓用权益的 frac 做保证金；notional = 合约市值为权益的 frac；risk = 每笔止损风险为权益的 frac
"""
import argparse
import os

import pandas as pd

from jm_strategy.backtest import Account, metrics, run_backtest, yearly_returns
from jm_strategy.data import get_jm_weighted, load_csv
from jm_strategy.library import backtest_params, base_series, build_signals

S4_PARAMS = dict(sar="s1", filt="stm4", exit="sar:slow", stop="atr2.5")


def streaks(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def core_stats(eq, tr, capital):
    m = metrics(eq, tr, capital)
    prev_eq = eq.shift(1).fillna(capital)
    tr = tr.copy()
    tr["权益占比"] = [t.pnl / prev_eq[t.entry_date] for t in tr.itertuples()]
    tr["名义杠杆"] = [t.lots * t.entry * 60 / prev_eq[t.entry_date] for t in tr.itertuples()]
    win, loss = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq[:trough].idxmax()
    rows = [
        ("初始资金", f"{capital:,.0f} 元"),
        ("期末权益", f"{eq.iloc[-1]:,.0f} 元"),
        ("总收益率", f"{m['总收益率']:.1%}"),
        ("年化收益率", f"{m['年化收益率']:.1%}"),
        ("最大回撤", f"{m['最大回撤']:.1%}（{peak.date()} 至 {trough.date()}，"
                    f"{eq[peak] - eq[trough]:,.0f} 元）"),
        ("卡玛比率", f"{m['卡玛比率']:.2f}"),
        ("夏普比率", f"{m['夏普比率']:.2f}"),
        ("交易次数", f"{len(tr)} 笔（年均 {m['年均交易次数']:.1f} 笔；多 {(tr.side == '多').sum()} / 空 {(tr.side == '空').sum()}）"),
        ("胜率", f"{len(win) / len(tr):.1%}（盈 {len(win)} 笔 / 亏 {len(loss)} 笔）"),
        ("盈亏比（均盈/均亏）", f"{win.pnl.mean() / -loss.pnl.mean():.2f}"),
        ("平均盈利 / 平均亏损", f"{win.pnl.mean():,.0f} 元 / {loss.pnl.mean():,.0f} 元"),
        ("平均盈利 / 平均亏损（占当时权益）", f"{win['权益占比'].mean():+.1%} / {loss['权益占比'].mean():+.1%}"),
        ("单笔最大盈利", f"{win.pnl.max():,.0f} 元（{win['权益占比'].max():+.1%}）"),
        ("单笔最大亏损", f"{loss.pnl.min():,.0f} 元（{loss['权益占比'].min():+.1%}）"),
        ("利润因子（总盈利/总亏损）", f"{win.pnl.sum() / -loss.pnl.sum():.2f}"),
        ("每笔平均期望", f"{tr.pnl.mean():,.0f} 元"),
        ("最大连续亏损 / 盈利", f"{streaks(tr.pnl <= 0)} 笔 / {streaks(tr.pnl > 0)} 笔"),
        ("平均持仓", f"{tr.bars.mean():.0f} 个交易日（盈利单 {win.bars.mean():.0f}，亏损单 {loss.bars.mean():.0f}）"),
        ("离场方式", "，".join(f"{k} {v} 笔" for k, v in tr.reason.value_counts().items())),
        ("平均手数 / 名义杠杆", f"{tr.lots.mean():.0f} 手 / {tr['名义杠杆'].mean():.2f} 倍（最高 {tr['名义杠杆'].max():.2f} 倍）"),
    ]
    for side, g in tr.groupby("side"):
        rows.append((f"{side}单", f"{len(g)} 笔，胜率 {(g.pnl > 0).mean():.0%}，合计 {g.pnl.sum():,.0f} 元"))
    return rows, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--capital", type=float, default=500_000)
    ap.add_argument("--sizing", default="margin", choices=["margin", "notional", "risk"])
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--margin-rate", type=float, default=0.15)
    ap.add_argument("--no-compound", action="store_true", help="始终按初始资金计算手数")
    ap.add_argument("--fee-rate", type=float, default=1.5e-4, help="每边手续费率（按成交额）")
    ap.add_argument("--fee-per-lot", type=float, default=0.0, help="每边每手另加的固定手续费，如交易所加一分钱 = 0.01")
    ap.add_argument("--fee-rate-today", type=float, default=None, help="平今费率，默认与 --fee-rate 相同")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load_csv(a.csv) if a.csv else get_jm_weighted()
    sig = build_signals(base_series(df), "S4", S4_PARAMS)
    acct = Account(capital=a.capital, sizing=a.sizing, pos_frac=a.frac, margin_rate=a.margin_rate,
                   risk_per_trade=a.frac, compound=not a.no_compound, fee_rate=a.fee_rate,
                   fee_per_lot=a.fee_per_lot, fee_rate_today=a.fee_rate_today)
    eq, tr, _ = run_backtest(df, backtest_params(sig), acct, sig=sig)
    rows, tr = core_stats(eq, tr, a.capital)
    print(f"S4 回测：初始资金 {a.capital:,.0f}，仓位方式 {a.sizing}，比例 {a.frac:.0%}"
          + (f"，保证金比例 {a.margin_rate:.0%}" if a.sizing == "margin" else "")
          + ("，不复利" if a.no_compound else "，复利"))
    for k, v in rows:
        print(f"  {k}：{v}")
    yr = yearly_returns(eq)
    year_end = eq.resample("YE").last()
    print("  年度收益：")
    for (y, r), e in zip(yr.items(), year_end):
        print(f"    {y}  {r:+7.1%}   年末权益 {e:,.0f}")
    for tag, rng in (("样本内 2013–2020", dict(end="2020-12-31")), ("样本外 2021–2026", dict(start="2021-01-01"))):
        e2, t2, _ = run_backtest(df, backtest_params(sig), acct, sig=sig, **rng)
        m2 = metrics(e2, t2, a.capital)
        w2, l2 = t2[t2.pnl > 0], t2[t2.pnl <= 0]
        print(f"  {tag}：年化 {m2['年化收益率']:.1%}，回撤 {m2['最大回撤']:.1%}，{len(t2)} 笔，"
              f"胜率 {len(w2) / len(t2):.1%}，盈亏比 {w2.pnl.mean() / -l2.pnl.mean():.2f}")
    os.makedirs(a.out, exist_ok=True)
    tag = f"s4_{int(a.capital)}_{a.sizing}{int(a.frac * 100)}" + ("_fixed" if a.no_compound else "")
    tr.to_csv(os.path.join(a.out, f"{tag}_trades.csv"), index=False, encoding="utf-8-sig")
    eq.to_csv(os.path.join(a.out, f"{tag}_equity.csv"))


if __name__ == "__main__":
    main()
