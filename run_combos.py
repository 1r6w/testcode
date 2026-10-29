"""四个指标 15 种组合 × 离场方式 × 趋势过滤 × 止损倍数 的对比回测。

离场方式：任一转向 / 全部反向 / ATR 吊灯(4ATR，所有组合通用) / 慢 SAR(仅含指标 1 的组合)

用法：python run_combos.py --csv data/JM_weighted_user.csv [--risk 0.02]
输出：results/combos.csv，并打印排名。
"""
import argparse
import itertools
import os

import pandas as pd

from jm_strategy.backtest import Account, metrics, run_backtest
from jm_strategy.combo import ALL_COMBOS, combo_name, combo_signals, indicator_states
from jm_strategy.data import get_jm_weighted, load_csv
from jm_strategy.strategy import Params

TRAIN_END, TEST_START = "2020-12-31", "2021-01-01"
EXITS = {"any": "任一转向", "all": "全部反向", "chand": "ATR吊灯4倍", "sarslow": "慢SAR"}
KEYS = ("年化收益率", "最大回撤", "卡玛比率", "交易次数", "胜率", "盈亏比(均盈/均亏)", "盈亏比(按R)", "期望值R/笔")


def run_all(df, acct):
    cache = indicator_states(df)
    rows = []
    for combo, exit_rule, trend_n, stop_k in itertools.product(ALL_COMBOS, EXITS, (0, 60), (1.5, 2.5)):
        if len(combo) == 1 and exit_rule == "all":
            continue  # 单指标两种离场等价
        if exit_rule == "sarslow" and 1 not in combo:
            continue  # 慢 SAR 属于指标 1
        sig = combo_signals(df, combo, exit_rule, trend_n, cache)
        p = Params(exit_mode="signal", stop_k=stop_k, trail_k=4.0 if exit_rule == "chand" else 0.0)
        row = dict(组合=combo_name(combo), 指标数=len(combo), 离场=EXITS[exit_rule],
                   MA60过滤="是" if trend_n else "否", 止损ATR=stop_k)
        for tag, rng in (("全", {}), ("内", dict(end=TRAIN_END)), ("外", dict(start=TEST_START))):
            eq, tr, _ = run_backtest(df, p, acct, sig=sig, **rng)
            m = metrics(eq, tr, acct.capital)
            row.update({f"{tag}_{k}": m.get(k) for k in KEYS})
        row["达标"] = (row["全_胜率"] or 0) >= 0.40 and 3 <= (row["全_盈亏比(均盈/均亏)"] or 0) <= 6
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--risk", type=float, default=0.02)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load_csv(a.csv) if a.csv else get_jm_weighted()
    res = run_all(df, Account(risk_per_trade=a.risk))
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "combos.csv"), index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.unicode.east_asian_width", True)
    show = ["组合", "离场", "MA60过滤", "止损ATR", "全_年化收益率", "全_最大回撤", "全_卡玛比率", "全_交易次数",
            "全_胜率", "全_盈亏比(均盈/均亏)", "全_期望值R/笔", "内_期望值R/笔", "外_期望值R/笔", "外_胜率", "达标"]
    print(f"共 {len(res)} 个变体，同时满足胜率>=40% 且盈亏比 3~6 的：{int(res['达标'].sum())} 个\n")
    print("按全样本卡玛比率排名前 20：")
    print(res.sort_values("全_卡玛比率", ascending=False)[show].head(20).round(3).to_string(index=False))
    print("\n每种组合的最好变体（按全样本卡玛比率）：")
    best = res.sort_values("全_卡玛比率", ascending=False).groupby("组合", sort=False).head(1)
    print(best[show].round(3).to_string(index=False))
    save_best(df, res, a)


def save_best(df, res, a):
    """把全样本卡玛比率最高的变体的逐笔交易和资金曲线写入 results/。"""
    top = res.sort_values("全_卡玛比率", ascending=False).iloc[0]
    combo = tuple(int(x) for x in top["组合"].split(" ")[0].split("+"))
    exit_rule = {v: k for k, v in EXITS.items()}[top["离场"]]
    sig = combo_signals(df, combo, exit_rule, 60 if top["MA60过滤"] == "是" else 0)
    p = Params(exit_mode="signal", stop_k=top["止损ATR"], trail_k=4.0 if exit_rule == "chand" else 0.0)
    acct = Account(risk_per_trade=a.risk)
    eq, tr, _ = run_backtest(df, p, acct, sig=sig)
    tr.to_csv(os.path.join(a.out, "combo_best_trades.csv"), index=False, encoding="utf-8-sig")
    eq.to_csv(os.path.join(a.out, "combo_best_equity.csv"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax[0].plot(eq / acct.capital, lw=1.2)
    ax[0].set_title(f"JM weighted - indicators {'+'.join(map(str, combo))}, exit={exit_rule}, "
                    f"stop={top['止损ATR']}ATR, risk={a.risk:.0%} (x initial)")
    ax[1].fill_between(eq.index, eq / eq.cummax() - 1, 0, color="tab:red", alpha=.4)
    ax[1].set_title("drawdown")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "combo_best_equity.png"), dpi=110)
    print(f"\n最优变体 {top['组合']} 的逐笔交易与资金曲线已写入 {a.out}/combo_best_*")


if __name__ == "__main__":
    main()
