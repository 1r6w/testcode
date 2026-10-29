"""焦煤加权指数 SAR+BBI+STM 共振策略回测入口。

用法：
  python run_backtest.py                       # 用 akshare 下载合约并合成加权指数后回测
  python run_backtest.py --csv JM加权.csv       # 用通达信/文华导出的加权日线回测
  python run_backtest.py --optimize            # 样本内(<=2020)网格寻参，样本外(2021起)检验
"""
import argparse
import itertools
import os

import pandas as pd

from jm_strategy.backtest import Account, metrics, run_backtest, yearly_returns
from jm_strategy.data import get_jm_weighted, load_csv
from jm_strategy.strategy import Params

TRAIN_END, TEST_START = "2020-12-31", "2021-01-01"


def fmt(m):
    pct = {"总收益率", "年化收益率", "最大回撤", "胜率"}
    return "\n".join(f"  {k:<14}{v:.2%}" if k in pct else
                     f"  {k:<14}{v:.2f}" if isinstance(v, float) else f"  {k:<14}{v}"
                     for k, v in m.items())


def score(m):
    """寻参目标：满足胜率>=40%、盈亏比 3~6 的前提下，卡玛比率最高。"""
    if m.get("交易次数", 0) < 15:
        return -1e9
    pen = 0.0
    pen += max(0, 0.40 - m["胜率"]) * 20
    pen += max(0, 3 - m["盈亏比(均盈/均亏)"]) + max(0, m["盈亏比(均盈/均亏)"] - 6)
    return m["卡玛比率"] - pen


def optimize(df, acct):
    grid = dict(stop_k=[1.5, 2.0, 2.5, 3.0], trail_k=[0, 3.0, 4.0], sar_step=[0.01, 0.02],
                stm_min=[0, 10], use_top_filter=[True, False])
    rows = []
    for vals in itertools.product(*grid.values()):
        p = Params(**dict(zip(grid, vals)))
        eq, tr, _ = run_backtest(df, p, acct, end=TRAIN_END)
        m = metrics(eq, tr, acct.capital)
        rows.append({**dict(zip(grid, vals)), **m, "score": score(m)})
    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    best = res.iloc[0]
    return Params(**{k: (bool(best[k]) if k == "use_top_filter" else best[k]) for k in grid}), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--risk", type=float, default=0.02, help="每笔风险占权益比例，调节年化/回撤")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = load_csv(a.csv) if a.csv else get_jm_weighted(refresh=a.refresh)
    acct = Account(risk_per_trade=a.risk)
    os.makedirs(a.out, exist_ok=True)

    p = Params()
    if a.optimize:
        p, res = optimize(df, acct)
        res.to_csv(os.path.join(a.out, "grid_train.csv"), index=False)
        print("样本内最优参数:", p)

    for name, kw in [("全样本", {}), ("样本内", dict(end=TRAIN_END)), ("样本外", dict(start=TEST_START))]:
        eq, tr, _ = run_backtest(df, p, acct, **kw)
        print(f"\n===== {name} =====\n{fmt(metrics(eq, tr, acct.capital))}")
        if name == "全样本":
            full_eq, full_tr = eq, tr

    print("\n年度收益:\n", yearly_returns(full_eq).map("{:.2%}".format).to_string())
    full_tr.to_csv(os.path.join(a.out, "trades.csv"), index=False)
    full_eq.to_csv(os.path.join(a.out, "equity.csv"))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
        ax[0].plot(full_eq / acct.capital, lw=1.2)
        ax[0].set_title("JM weighted - SAR+BBI+STM equity (x initial)")
        ax[1].fill_between(full_eq.index, full_eq / full_eq.cummax() - 1, 0, color="tab:red", alpha=.4)
        ax[1].set_title("drawdown")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "equity.png"), dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
