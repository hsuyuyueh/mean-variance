import pandas as pd
import numpy as np
from datetime import datetime
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg

def backtest_simulator(prices: pd.DataFrame,
                       initial_capital: float = 1_000_000,
                       rebalance_freq: str = 'M',
                       rf_rate: float = 0.015):
    """
    prices: DataFrame, index=Date, columns=tickers, 欄位為收盤價
    rebalance_freq: 'M' 月、'2W' 兩週、'W' 週
    回傳 DataFrame: 每個 rebalancing 之後的 portfolio value
    """
    # 1. 產生再平衡日期
    dates = pd.date_range(start=prices.index[0], end=prices.index[-1], freq=rebalance_freq)
    nav = []               # 每期資產淨值
    nav.append({'date': dates[0], 'value': initial_capital})

    current_value = initial_capital
    for i in range(1, len(dates)):
        as_of = dates[i]
        # 2. 取 as_of 以前的資料，用來估 mu, sigma
        hist = prices[:as_of].dropna(axis=1, how='any')
        mu = np.log(hist / hist.shift(1)).mean() * 252       # 年化
        sigma = exp_cov(hist, span=180)

        # 3. 最適化
        ef = EfficientFrontier(mu.to_dict(), sigma, weight_bounds=(0,1))
        ef.add_objective(L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=rf_rate)
        w = ef.clean_weights()

        # 4. 計算這段期間的投組報酬
        next_period = prices.loc[as_of:dates[i], list(w.keys())].pct_change().dropna()
        # 投組每天報酬 = 各股當天報酬 * 權重
        port_rets = next_period.dot(pd.Series(w))
        # 將當期資產價值滾動
        current_value *= (1 + port_rets).cumprod().iloc[-1]
        nav.append({'date': dates[i], 'value': current_value})

    df_nav = pd.DataFrame(nav).set_index('date')
    df_nav['return'] = df_nav['value'].pct_change().fillna(0)
    # 計算績效指標
    total_ret = df_nav['value'].iloc[-1]/initial_capital - 1
    ann_ret = (1 + total_ret) ** (252/len(prices)) - 1
    vol = df_nav['return'].std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / vol

    print(f"Backtest 總報酬: {total_ret:.2%}, 年化報酬: {ann_ret:.2%}, 年化波動: {vol:.2%}, Sharpe: {sharpe:.2f}")
    return df_nav

# ===== 使用範例 =====
import yfinance as yf

tickers = ['2330.TW','2303.TW','2412.TW']  # 範例
prices = yf.download(tickers, start='2023-01-01', end='2025-07-15')['Adj Close']

df_nav = backtest_simulator(prices,
                            initial_capital=10_000_000,
                            rebalance_freq='M',  # 每月
                            rf_rate=0.015)

print(df_nav)
