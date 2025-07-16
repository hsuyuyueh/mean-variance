#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest_asyncio
import pandas as pd
import yfinance as yf
from pypfopt.risk_models import sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from fetch_0056_components import fetch_0056_components
from datetime import datetime, timedelta
from my_ai_module import gpt_contextual_rating
from tqdm import tqdm
from scipy.stats import norm

def get_results():
    return {
        "exp_ret": exp_ret,
        "vol": vol,
        "sharpe": sharpe,
        "weights": weights_final
    }

# 解決 asyncio 衝突，使同一 event loop 可以多次 re-enter
nest_asyncio.apply()


def load_prices(tickers, start, end):
    """
    下載調整後收盤價
    """
    data = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=False)
    return data['Adj Close']


def optimize_portfolio(mu, sigma, risk_free_rate=0.015,
                       lower_bound=0.0, upper_bound=1.0,
                       objective='max_sharpe', target_return=None,
                       target_volatility=None):
    """
    使用 PyPortfolioOpt 進行平均-變異數最佳化
    mu: dict of expected returns
    sigma: DataFrame of covariance
    返回 weights, performance
    """
    ef = EfficientFrontier(mu, sigma, weight_bounds=(lower_bound, upper_bound))
    if objective == 'max_sharpe':
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == 'min_volatility':
        ef.min_volatility()
    elif objective == 'efficient_return':
        ef.efficient_return(target_return)
    elif objective == 'efficient_risk':
        ef.efficient_risk(target_volatility)
    else:
        raise ValueError(f"不支援的目標：{objective}")

    weights = ef.clean_weights()
    perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
    return weights, perf


def recent_drawdown(prices, window=10):
    """
    計算最近 window 天的最大回撤比例，回傳單一標量
    """
    nav = (1 + prices.pct_change().dropna()).cumprod().iloc[-window:]
    peak = nav.cummax()
    drawdown = (peak - nav) / peak
    # 先計算每檔資產在窗口中的最大回撤，再取整體最大值
    max_dd_per_asset = drawdown.max()
    return max_dd_per_asset.max()


def apply_risk_controls(weights_raw, current_weights, prices,
                        cap=0.20, threshold=0.05, dd_limit=0.15, dd_leverage=0.8):
    """
    風控修正：
      1. 單檔權重上限 cap
      2. 與 current_weights 偏差小於 threshold 時維持原值
      3. 若短期回撤超過 dd_limit，全部乘以 dd_leverage
      4. 正規化權重之和為 1
    """
    # 單檔上限
    w_cap = {tk: min(w, cap) for tk, w in weights_raw.items()}
    # 偏差門檻
    w_adj = {}
    for tk, w in w_cap.items():
        current = current_weights.get(tk, 0)
        if abs(w - current) > threshold:
            w_adj[tk] = w
        else:
            w_adj[tk] = current
    # 極端事件降槓桿
    if recent_drawdown(prices) > dd_limit:
        w_adj = {tk: w * dd_leverage for tk, w in w_adj.items()}
    # 正規化
    total = sum(w_adj.values())
    if total > 0:
        w_final = {tk: w / total for tk, w in w_adj.items()}
    else:
        w_final = w_adj
    return w_final


def load_current_portfolio():
    """
    載入目前持倉權重，格式同 weights，如無資料全零
    """
    return {}


if __name__ == '__main__':
    # 1. 擷取成分股代號
    components = fetch_0056_components()
    tickers = [code for code, _ in components]

    # 2. 設定樣本期：過去一年
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    rf_rate, capital = 0.015, 10_000_000

    # 3. 下載價格
    prices = load_prices(tickers, start_date, end_date)
    sigma = sample_cov(prices)

    # 4. 使用 AI 模型預測報酬率 by horizon_months
    horizon_months=3
    mu_pred = gpt_contextual_rating(tickers, horizon_months=horizon_months)

    # 5. 原始最適化
    weights_raw, (exp_ret, vol, sharpe) = optimize_portfolio(
        mu_pred, sigma, risk_free_rate=rf_rate)

    # 6. 風控修正
    current_weights = load_current_portfolio()
    weights_final = apply_risk_controls(weights_raw, current_weights, prices)

    # 7. 顯示結果
    print(f"使用樣本期：{start_date} 到 {end_date}\n")
    print('=== 最佳化結果（Max Sharpe）===')
    probability = norm.cdf(sharpe)  # Φ(Sharpe)
    for tk, w in weights_final.items():
        if w > 0:
            print(f'  {tk}: {w:.2%}')
    print(f"\n預期年化報酬: {exp_ret:.2%}, 年化波動率: {vol:.2%}, Sharpe: {sharpe:.2f}, Probability : {probability:.2f} \n")
    print(f'=== 資金配置 (總資金 {capital:,} TWD) ===')
    for tk, w in weights_final.items():
        amount = capital * w
        if amount > 0:
            print(f'  {tk}: {amount:,.0f} TWD')

    # 8. 產生並執行交易指令
    # orders = compute_trade_orders(current_portfolio, weights_final)
    # execute_orders(orders)
