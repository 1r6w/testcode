"""动能拐点（用户思路）回测：单独使用、与 1+4 趋势策略混合，并与 1+4 对比。

用法：
  python run_turn.py --csv data/JM_weighted_user.csv           # 打印关键方案对比
  python run_turn.py --csv data/JM_weighted_user.csv --grids   # 另外重跑两轮网格（约 5 分钟），写入 results/
"""
import argparse
import itertools
import os

import pandas as pd

from jm_strategy.backtest import Account, metrics, run_backtest
from jm_strategy.combo import combo_signals, indicator_states
from jm_strategy.data import get_jm_weighted, load_csv
from jm_strategy.hybrid import HybridParams, hybrid_backtest_params, hybrid_signals
from jm_strategy.strategy import Params
from jm_strategy.turn import TurnParams, turn_backtest_params, turn_signals

PERIODS = (("全", {}), ("内", dict(end="2020-12-31")), ("外", dict(start="2021-01-01")))
COLS = (("交易次数", "n"), ("胜率", "win"), ("盈亏比(均盈/均亏)", "pr"), ("期望值R/笔", "E"),
        ("年化收益率", "cagr"), ("最大回撤", "dd"), ("卡玛比率", "calmar"))


def evaluate(df, sig, p, acct):
    row = {}
    for tag, rng in PERIODS:
        eq, tr, _ = run_backtest(df, p, acct, sig=sig, **rng)
        m = metrics(eq, tr, acct.capital)
        row.update({f"{tag}_{short}": m.get(key) for key, short in COLS})
    return row


def turn_grid(df, acct):
    grid = dict(k=[2, 3, 4], hh_n=[0, 1, 2, 3], ind4=["all", "last", "off"], depth=[0, 20, 40],
                trend=["none", "with", "against"], stop_mode=["swing", "atr"],
                exit=["turn", "slow", "turn+slow", "chand", "turn+chand"], breakeven_r=[0, 1.5])
    rows = []
    for vals in itertools.product(*grid.values()):
        tp = TurnParams(**dict(zip(grid, vals)))
        rows.append({**dict(zip(grid, vals)), **evaluate(df, turn_signals(df, tp), turn_backtest_params(tp), acct)})
    return pd.DataFrame(rows)


def hybrid_grid(df, acct):
    cache = indicator_states(df)
    grid = dict(mode=["exit", "entry", "both"], k=[2, 3, 4], hh_n=[1, 2, 3], ind4=["all", "last", "off"],
                turn_trend=["none", "with"], turn_stop=["swing", "atr"], turn_exit=["turn", "slow", "turn+slow"])
    rows, seen = [], set()
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        if kw["mode"] == "exit":  # 只提前平仓时，拐点开仓参数不起作用
            key = (kw["k"], kw["hh_n"], kw["ind4"])
            if key in seen:
                continue
            seen.add(key)
            hp = HybridParams(mode="exit", k=kw["k"], hh_n=kw["hh_n"], ind4=kw["ind4"])
            kw.update(turn_trend="-", turn_stop="-", turn_exit="-")
        else:
            hp = HybridParams(**kw)
        rows.append({**kw, **evaluate(df, hybrid_signals(df, hp, cache), hybrid_backtest_params(hp), acct)})
    return pd.DataFrame(rows)


def key_cases(df):
    cache = indicator_states(df)
    raw = TurnParams()
    best_turn = TurnParams(k=2, hh_n=3, ind4="off", stop_mode="atr", exit="slow")
    hp_f = HybridParams(mode="entry", k=2, hh_n=3, ind4="last", turn_trend="with", turn_stop="atr", turn_exit="turn+slow")
    hp_g = HybridParams(mode="entry", k=2, hh_n=3, ind4="off", turn_trend="with", turn_stop="atr", turn_exit="turn+slow")
    hp = hybrid_backtest_params(HybridParams())
    return [
        ("A 1+4 趋势（现有）", combo_signals(df, (1, 4), "sarslow", 60, cache), Params(exit_mode="signal", stop_k=2.5)),
        ("B 拐点原始思路（单独）", turn_signals(df, raw), turn_backtest_params(raw)),
        ("C 拐点单独-样本内最优", turn_signals(df, best_turn), turn_backtest_params(best_turn)),
        ("D 1+4 + 平仓抄底/平仓开空", hybrid_signals(df, HybridParams(mode="both"), cache), hp),
        ("E 1+4 + 拐点提前止盈", hybrid_signals(df, HybridParams(mode="exit"), cache), hp),
        ("F 混合-样本内最优", hybrid_signals(df, hp_f, cache), hp),
        ("G 混合-三段都好（事后挑选）", hybrid_signals(df, hp_g, cache), hp),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--grids", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load_csv(a.csv) if a.csv else get_jm_weighted()
    acct = Account()
    os.makedirs(a.out, exist_ok=True)
    if a.grids:
        turn_grid(df, acct).to_csv(os.path.join(a.out, "turn_grid.csv"), index=False, encoding="utf-8-sig")
        hybrid_grid(df, acct).to_csv(os.path.join(a.out, "hybrid_grid.csv"), index=False, encoding="utf-8-sig")
    rows = [{"方案": name, **evaluate(df, sig, p, acct)} for name, sig, p in key_cases(df)]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "turn_compare.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    show = ["方案", "全_cagr", "全_dd", "全_calmar", "全_n", "全_win", "全_pr", "外_cagr", "外_calmar", "外_win", "外_pr"]
    print(res[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
