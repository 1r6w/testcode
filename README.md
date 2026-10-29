# 焦煤加权 SAR + BBI + STM 三重共振趋势策略

四个指标都用上了，各司其职：

| 指标 | 在策略中的角色 |
|---|---|
| 指标一 SAR(10,2,20) | 趋势方向 + **移动止盈**（SAR 翻转离场） |
| 指标二 BBI(3,6,12,24) | 价格结构过滤：多单要求收盘 > BBI，空单要求收盘 < BBI |
| 指标三 STM（ATR 标准化综合动能） | 动能方向：STM>0 做多 / STM<0 做空；**顶部预警**当根不追多 |
| 指标四 STM 简化版 | 未做 ATR 标准化，数值随价格水平漂移，2013 年 700 元与 2021 年 3800 元的焦煤不可比，因此只实现、不参与交易（`indicators.ind_stm_v2`） |

## 交易规则（日线，焦煤加权指数）

- **开多**：STM>0、收盘>BBI、SAR 多头，三者从“不全满足”变成“全满足”的那根 K 线收盘出信号，次日开盘进场；当根出现顶部预警不进。
- **开空**：完全对称（底部预警当根不进）。
- **初始止损**：入场价 ∓ `stop_k` × ATR(14)，盘中触发，跳空按开盘价成交。
- **止盈/离场**：SAR 翻转（收盘确认，次日开盘平）；可选吊灯止损 `trail_k` × ATR。平仓当根若反向共振已成立则直接反手。
- **仓位**：每笔初始止损风险 = 权益 × `risk`（默认 2%），名义杠杆上限 3 倍。年化收益和回撤都与 `risk` 近似线性相关，是调节年化的主要旋钮。
- **成本**：手续费双边各 1.5‱ 成交额 + 每边 1 跳（0.5 元/吨）滑点，合约乘数 60。

设计思路是“小亏大赚”：亏损单被固定在约 1R，盈利单由 SAR 跟随趋势，用来拉高盈亏比；
三重共振过滤掉一部分震荡假突破，用来托住胜率。

## 运行

```bash
pip install akshare pandas matplotlib
python run_backtest.py --optimize            # 自动下载 JM 全部合约，按持仓量合成加权指数
python run_backtest.py --csv JM加权日线.csv    # 或使用通达信 JML9 / 文华 JM888 导出的日线
python run_backtest.py --optimize --risk 0.03 # 调整单笔风险以调节年化
```

`--optimize` 只在 2013–2020 样本内寻参（目标：胜率 ≥ 40%、盈亏比 3–6 的约束下卡玛比率最高），
2021 年以后作为样本外检验，避免把参数拟合到全部历史上。
输出：`results/trades.csv`（逐笔）、`results/equity.csv`、`results/equity.png`、`results/grid_train.csv`。

## 通达信交易系统公式（与 Python 版逻辑一致，可直接在软件里做加权回测）

参数：STOPK=2, STEP=2, MVALUE=20

```
STEP1:=STEP/100; MVALUE1:=MVALUE/100;
SARV:=SAR(10,STEP1,MVALUE1);
SARUP:=CLOSE>SARV;
BBIV:=(MA(C,3)+MA(C,6)+MA(C,12)+MA(C,24))/4;
TR1:=MAX(HIGH-LOW,MAX(ABS(HIGH-REF(C,1)),ABS(LOW-REF(C,1))));
A:=MAX(MA(TR1,14),0.01);
T1:=(MA(C,5)-MA(C,20))/A*20;
DIF1:=EMA(C,12)-EMA(C,26); DEA1:=EMA(DIF1,9);
M1:=(DIF1-DEA1)/A*10;
RSI1:=SMA(MAX(C-REF(C,1),0),14,1)/MAX(SMA(ABS(C-REF(C,1)),14,1),0.01)*100;
STM:=EMA(T1+M1+(RSI1-50)*0.5,3);
顶部预警:=C>REF(HHV(H,10),1) AND STM<REF(STM,1) AND STM>0;
底部预警:=C<REF(LLV(L,10),1) AND STM>REF(STM,1) AND STM<0;
多共振:=STM>0 AND C>BBIV AND SARUP;
空共振:=STM<0 AND C<BBIV AND NOT(SARUP);
BK:=多共振 AND NOT(REF(多共振,1)) AND NOT(顶部预警);
SK:=空共振 AND NOT(REF(空共振,1)) AND NOT(底部预警);
SP:CROSS(SARV,C);
BP:CROSS(C,SARV);
BK; SK;
{初始止损在交易系统设置中填 STOPK*ATR，或文华用 ENTRYPRICE-STOPK*A 写止损}
```
