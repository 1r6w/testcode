"""策略库回测：S1~S7 每个策略在自己的参数网格上用 2013–2020 选参、2021–2026 检验，并与 1+4 基准对比；
再把 1+4、S4、S3 三个策略分仓组合（资金三等分）。

用法：python run_library.py --csv data/JM_weighted_user.csv
输出：results/library_all.csv（全部变体）、results/library_best.csv（每个策略的入选变体及组合）、
     results/library_equity.png（资金曲线）、results/library_calmar.png（卡玛比率对比）
"""
import argparse
import ast
import os

import pandas as pd

from jm_strategy.backtest import Account, metrics, run_backtest
from jm_strategy.combo import combo_signals, indicator_states
from jm_strategy.data import get_jm_weighted, load_csv
from jm_strategy.library import STRATEGIES, backtest_params, base_series, build_signals, variants
from jm_strategy.strategy import Params

PORTFOLIO = ("B0", "S4", "S3")

PERIODS = (("全", {}), ("内", dict(end="2020-12-31")), ("外", dict(start="2021-01-01")))
COLS = (("交易次数", "n"), ("胜率", "win"), ("盈亏比(均盈/均亏)", "pr"), ("期望值R/笔", "E"),
        ("年化收益率", "cagr"), ("最大回撤", "dd"), ("卡玛比率", "calmar"))
MIN_IS_TRADES = 15


def evaluate(df, sig, acct, p=None):
    p, row = p or backtest_params(sig), {}
    for tag, rng in PERIODS:
        eq, tr, _ = run_backtest(df, p, acct, sig=sig, **rng)
        m = metrics(eq, tr, acct.capital)
        row.update({f"{tag}_{short}": m.get(key) for key, short in COLS})
    return row


def run_all(df, acct):
    b = base_series(df)
    rows = []
    for code, (name, _, _) in STRATEGIES.items():
        for p in variants(code):
            sig = build_signals(b, code, p)
            rows.append({"策略": code, "名称": name, "参数": str(p), **evaluate(df, sig, acct)})
    return pd.DataFrame(rows)


def pick_best(res):
    """每个策略按样本内卡玛比率选一个变体（样本内至少 15 笔），并统计整个参数族的样本外表现。"""
    out = []
    for code, g in res.groupby("策略", sort=False):
        cand = g[g["内_n"] >= MIN_IS_TRADES]
        best = (cand if len(cand) else g).sort_values("内_calmar", ascending=False).iloc[0].to_dict()
        best["族_变体数"] = len(g)
        best["族_外卡玛中位"] = g["外_calmar"].median()
        best["族_外正期望占比"] = (g["外_E"] > 0).mean()
        out.append(best)
    return pd.DataFrame(out)


def benchmark(df):
    return combo_signals(df, (1, 4), "sarslow", 60, indicator_states(df)), Params(exit_mode="signal", stop_k=2.5)


def legs(df, best):
    """返回 {代码: (信号, 回测参数)}，含基准与每个策略的入选变体。"""
    b, out = base_series(df), {"B0": benchmark(df)}
    for _, r in best[best["策略"].str.startswith("S")].iterrows():
        sig = build_signals(b, r["策略"], ast.literal_eval(r["参数"]))
        out[r["策略"]] = (sig, backtest_params(sig))
    return out


def portfolio_equity(df, leg_map, codes, acct, **rng):
    return sum(run_backtest(df, leg_map[c][1], acct, sig=leg_map[c][0], **rng)[0] for c in codes) / len(codes)


def portfolio_row(df, leg_map, acct):
    row = {"策略": "组合", "名称": "1+4 / S4 / S3 资金三等分", "参数": str(PORTFOLIO)}
    for tag, rng in PERIODS:
        m = metrics(portfolio_equity(df, leg_map, PORTFOLIO, acct, **rng), pd.DataFrame(), acct.capital)
        row.update({f"{tag}_cagr": m["年化收益率"], f"{tag}_dd": m["最大回撤"], f"{tag}_calmar": m["卡玛比率"]})
    return row


def plot(df, leg_map, best, acct, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

    def style(ax):
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=9)
        ax.grid(color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    series = [("1+4 趋势（基准）", run_backtest(df, leg_map["B0"][1], acct, sig=leg_map["B0"][0])[0]),
              ("S4 灵敏SAR", run_backtest(df, leg_map["S4"][1], acct, sig=leg_map["S4"][0])[0]),
              ("S3 绿变红", run_backtest(df, leg_map["S3"][1], acct, sig=leg_map["S3"][0])[0]),
              ("三策略组合", portfolio_equity(df, leg_map, PORTFOLIO, acct))]
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=surface)
    style(ax)
    for (name, eq), col in zip(series, colors):
        y = eq / acct.capital
        ax.plot(y.index, y, color=col, linewidth=2, label=name)
        ax.annotate(f"{name} {y.iloc[-1]:.2f}倍", (y.index[-1], y.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ink)
    ax.axvline(pd.Timestamp("2021-01-01"), color=ink2, linewidth=1, linestyle="--")
    ax.text(pd.Timestamp("2021-02-01"), ax.get_ylim()[1], "样本外 →", va="top", fontsize=9, color=ink2)
    ax.set_title("焦煤加权 2013-03-22 ~ 2026-10-29 资金曲线（单笔风险 2%，初始资金 = 1）", color=ink, fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink)
    ax.set_xlim(right=ax.get_xlim()[1] + 700)
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2014, 2027, 2)], [str(y) for y in range(2014, 2027, 2)])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "library_equity.png"), dpi=120, facecolor=surface)

    tab = best.set_index("策略")
    order = ["B0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "组合"]
    labels = [f"{c} {tab.loc[c, '名称']}" for c in order]
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=surface)
    style(ax)
    ypos = range(len(order))
    h = 0.38
    ax.barh([y - h / 2 - 0.01 for y in ypos], [tab.loc[c, "内_calmar"] for c in order], height=h, color=colors[0],
            label="样本内 2013–2020")
    ax.barh([y + h / 2 + 0.01 for y in ypos], [tab.loc[c, "外_calmar"] for c in order], height=h, color=colors[1],
            label="样本外 2021–2026")
    for y, c in zip(ypos, order):
        v = tab.loc[c, "外_calmar"]
        ax.text(max(v, 0) + 0.02, y + h / 2, f"{v:.2f}", va="center", fontsize=8, color=ink)
    ax.axvline(0, color=ink2, linewidth=1)
    ax.set_yticks(list(ypos), labels, fontsize=9, color=ink)
    ax.invert_yaxis()
    ax.set_title("各策略卡玛比率（年化 ÷ 最大回撤），每个策略按样本内选参", color=ink, fontsize=12, loc="left", pad=28)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "library_calmar.png"), dpi=120, facecolor=surface)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    df = load_csv(a.csv) if a.csv else get_jm_weighted()
    acct = Account()
    res = run_all(df, acct)
    best = pick_best(res)
    sig0, p0 = benchmark(df)
    b0 = {"策略": "B0", "名称": "1+4趋势共振(基准)", "参数": "1+4, MA60, 慢SAR离场, 2.5ATR",
          **evaluate(df, sig0, acct, p0)}
    best = pd.concat([pd.DataFrame([b0]), best], ignore_index=True)
    leg_map = legs(df, best)
    best = pd.concat([best, pd.DataFrame([portfolio_row(df, leg_map, acct)])], ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "library_all.csv"), index=False, encoding="utf-8-sig")
    best.to_csv(os.path.join(a.out, "library_best.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.unicode.east_asian_width", True)
    show = ["策略", "名称", "全_cagr", "全_dd", "全_calmar", "全_n", "全_win", "全_pr", "内_calmar",
            "外_calmar", "外_win", "外_pr", "族_外卡玛中位", "族_外正期望占比"]
    print(best[show].round(3).to_string(index=False))
    print()
    print(best[["策略", "参数"]].to_string(index=False))
    plot(df, leg_map, best, acct, a.out)
    print("\n三策略组合 单笔风险与年化/回撤：")
    for risk in (0.02, 0.04, 0.06, 0.08, 0.10):
        ra = Account(risk_per_trade=risk, max_leverage=5.0)
        parts = []
        for tag, rng in (("全", {}), ("外", dict(start="2021-01-01"))):
            m = metrics(portfolio_equity(df, leg_map, PORTFOLIO, ra, **rng), pd.DataFrame(), ra.capital)
            parts.append(f"{tag} 年化 {m['年化收益率']:.1%} 回撤 {m['最大回撤']:.1%} 卡玛 {m['卡玛比率']:.2f}")
        print(f"  {risk:.0%}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
