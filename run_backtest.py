"""焦煤加权指数 SAR+BBI+STM 共振策略回测入口。

用法：
  python run_backtest.py                       # 用 akshare 下载合约并合成加权指数后回测
  python run_backtest.py --csv JM加权.csv       # 用通达信/文华导出的加权日线回测
  python run_backtest.py --csv data/JM_weighted_user.csv --grid   # 附带参数邻域稳健性检验
  python run_backtest.py --csv data/JM_weighted_user.csv --risk 0.05  # 调整单笔风险
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


def robustness_grid(df, acct):
    """参数邻域稳健性检验：离场 SAR 步长 × 初始止损 × 趋势均线 × STM 门槛，共 160 组，分样本内/样本外统计。"""
    grid = dict(exit_sar_step=[0.004, 0.005, 0.0075, 0.01], stop_k=[1.0, 1.25, 1.5, 1.75, 2.0],
                trend_n=[40, 60, 80, 100], stm_min=[0, 10])
    rows = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        p = Params(exit_sar_max=kw["exit_sar_step"] * 10, **kw)
        row = dict(kw)
        for tag, rng in (("IS", dict(end=TRAIN_END)), ("OOS", dict(start=TEST_START))):
            eq, tr, _ = run_backtest(df, p, acct, **rng)
            m = metrics(eq, tr, acct.capital)
            row.update({f"{tag}_{k}": m.get(k) for k in ("交易次数", "胜率", "盈亏比(按R)", "期望值R/笔", "年化收益率", "最大回撤")})
        rows.append(row)
    res = pd.DataFrame(rows)
    print(f"\n参数邻域 {len(res)} 组：样本内正期望占比 {(res['IS_期望值R/笔'] > 0).mean():.1%}，"
          f"样本外正期望占比 {(res['OOS_期望值R/笔'] > 0).mean():.1%}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--grid", action="store_true", help="输出参数邻域稳健性检验")
    ap.add_argument("--risk", type=float, default=0.02, help="每笔风险占权益比例，调节年化/回撤")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = load_csv(a.csv) if a.csv else get_jm_weighted(refresh=a.refresh)
    acct = Account(risk_per_trade=a.risk)
    os.makedirs(a.out, exist_ok=True)

    p = Params()
    print("策略参数:", p)
    if a.grid:
        robustness_grid(df, acct).to_csv(os.path.join(a.out, "robustness_grid.csv"), index=False)

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
