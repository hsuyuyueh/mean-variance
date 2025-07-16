#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest_asyncio
import pandas as pd
import yfinance as yf
import json
import os
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg
from fetch_0056_components import fetch_0056_components
from fetch_0050_components import fetch_0050_components
from datetime import datetime, timedelta
from my_ai_module import gpt_contextual_rating
from tqdm import tqdm
from scipy.stats import norm

nest_asyncio.apply()

CURRENT_FILE = "current_portfolio.json"

# === 基本模組 ===
def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    return data['Adj Close']

def apply_momentum_adjustment(mu_series, prices, window=60, weight=0.3):
    """
    根據近 window 天動能調整 mu 值，加入動能強弱作為加權修正
    """
    momentum = (prices.iloc[-1] / prices.iloc[-window] - 1).fillna(0)
    adjusted_mu = mu_series * (1 + weight * momentum)
    return adjusted_mu

def apply_mu_confidence_adjustment(mu_series, prices, window=60, penalty=0.5):
    volatility = prices.pct_change().rolling(window).std().mean().fillna(0)
    scaled = 1 / (1 + penalty * volatility)
    adjusted_mu = mu_series * scaled
    return adjusted_mu

def compute_market_risk_factor(prices, dd_limit=0.15):
    try:
        vix = yf.download("^VIX", period="30d", interval="1d")['Close'].mean()
    except:
        vix = 20
    try:
        jnk = yf.download("JNK", period="30d", interval="1d")['Close'].pct_change().mean()
    except:
        jnk = 0
    drawdown = recent_drawdown(prices, window=10)
    vix_factor = max(0.4, 1.0 - (vix - 20) / 40)
    jnk_factor = max(0.4, 1.0 + jnk * 8)
    drawdown_factor = 0.7 if drawdown > dd_limit else 1.0
    return max(0.4, min(1.0, vix_factor * jnk_factor * drawdown_factor))

def optimize_portfolio(mu, sigma, risk_free_rate=0.015, lower_bound=0.0, upper_bound=1.0,
                       objective='max_sharpe', target_return=None, target_volatility=None,
                       regularize=False, gamma=0.1):
    ef = EfficientFrontier(mu, sigma, weight_bounds=(lower_bound, upper_bound))
    if regularize:
        ef.add_objective(L2_reg, gamma=gamma)
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
    nav = (1 + prices.pct_change().dropna()).cumprod().iloc[-window:]
    peak = nav.cummax()
    drawdown = (peak - nav) / peak
    return drawdown.max().max()

def apply_risk_controls(weights_raw, current_weights, prices,
                        cap=0.20, threshold=0.05, dd_limit=0.15, dd_leverage=0.8,
                        apply_market_risk=True):
    w_cap = {tk: min(w, cap) for tk, w in weights_raw.items()}
    w_adj = {}
    for tk, w in w_cap.items():
        current = current_weights.get(tk, 0)
        w_adj[tk] = w if abs(w - current) > threshold else current
    if recent_drawdown(prices) > dd_limit:
        w_adj = {tk: w * dd_leverage for tk, w in w_adj.items()}

    # v4 強化：整體曝險比例動態調整
    if apply_market_risk:
        exposure_factor = compute_market_risk_factor(prices)
        w_adj = {tk: w * exposure_factor for tk, w in w_adj.items()}

    total = sum(w_adj.values())
    return {tk: w / total for tk, w in w_adj.items()} if total > 0 else w_adj

def fetch_fundamentals(tickers):
    fundamentals = {}
    for tk in tqdm(tickers, desc="抓取基本面資料"):
        try:
            info = yf.Ticker(tk).info
            fundamentals[tk] = {
                'pe': info.get('trailingPE', 20),
                'roe': info.get('returnOnEquity', 0.1) * 100
            }
        except:
            fundamentals[tk] = {'pe': 20, 'roe': 10}
    return pd.DataFrame.from_dict(fundamentals, orient='index')

def apply_style_adjustment(mu, fundamentals):
    pe_scaled = 1 / (fundamentals['pe'] + 5)
    roe_scaled = fundamentals['roe'] / 20
    adj_factor = 0.5 * pe_scaled + 0.5 * roe_scaled
    return mu * adj_factor

def load_current_portfolio(filepath=CURRENT_FILE):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def save_current_portfolio(weights, filepath=CURRENT_FILE):
    with open(filepath, 'w') as f:
        json.dump(weights, f, indent=2)

def print_rebalance_plan(old, new):
    print("\n=== 建議調整持股變動 ===")
    for tk in sorted(set(old) | set(new)):
        before = old.get(tk, 0)
        after = new.get(tk, 0)
        diff = after - before
        if abs(diff) > 0.01:
            action = "加碼" if diff > 0 else "減碼"
            print(f"{action} {tk}: {diff:+.2%}")

if __name__ == '__main__':
    components_0050 = fetch_0050_components()
    components_0056 = fetch_0056_components()
    combined_components = {code: name for code, name in components_0050 + components_0056}
    tickers = [tk for tk in combined_components.keys() if not tk.startswith("289")]

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    irx = yf.download("^IRX", period="7d", interval="1d")['Close'].mean() / 100
    rf_rate, capital = irx if irx else 0.015, 10_000_000

    prices = load_prices(tickers, start_date, end_date)
    sigma = exp_cov(prices, span=180)

    mu_base = gpt_contextual_rating(tickers, horizon_months=3)
    fundamentals = fetch_fundamentals(tickers)
    mu_style = apply_style_adjustment(mu_base, fundamentals)
    mu_momentum = apply_momentum_adjustment(mu_style, prices, window=60, weight=0.3)
    mu_final = apply_mu_confidence_adjustment(mu_momentum, prices)

    weights_raw, (exp_ret, vol, sharpe) = optimize_portfolio(
        mu_final, sigma, risk_free_rate=rf_rate, regularize=True, gamma=0.1)

    current_weights = load_current_portfolio()
    weights_final = apply_risk_controls(weights_raw, current_weights, prices)

    print(f"使用樣本期：{start_date} 到 {end_date}\n")
    print('=== 最佳化結果（v6: AI μ + Style + Momentum + Risk Filter）===')
    probability = norm.cdf(sharpe)
    for tk, w in sorted(weights_final.items(), key=lambda x: -x[1]):
        if w > 0:
            mu_val = mu_final.get(tk, 0)
            try:
                sector = yf.Ticker(tk).info.get('sector', 'N/A')
            except:
                sector = 'N/A'
            print(f"  {tk}: {w:.2%}, μ={mu_val:.2%}, Sector={sector}")

    print(f"\n預期年化報酬: {exp_ret:.2%}, 年化波動率: {vol:.2%}, Sharpe: {sharpe:.2f}, P: {probability:.2f}")

    print_rebalance_plan(current_weights, weights_final)
    save_current_portfolio(weights_final)



# 🔄 回測模組（支援雙週再平衡）
def backtest_simulator(prices, tickers, horizon_months=3, rebalance_freq='2W', mu_cache=None, fundamentals_cache=None):
    import numpy as np
    from dateutil.rrule import rrule, WEEKLY

    results = []
    rebalance_days = pd.date_range(start=prices.index[0], end=prices.index[-1], freq=rebalance_freq)

    for rebalance_date in rebalance_days:
        if rebalance_date not in prices.index:
            continue
        window_prices = prices[:rebalance_date].dropna(axis=1, how='any')
        if len(window_prices) < 100:
            continue

        sigma = exp_cov(window_prices, span=180)

                        if mu_cache is None:
            mu_base = gpt_contextual_rating(tickers, horizon_months=horizon_months)
        else:
            mu_base = mu_cache
                        if fundamentals_cache is None:
            fundamentals = fetch_fundamentals(tickers)
        else:
            fundamentals = fundamentals_cache
        mu_style = apply_style_adjustment(mu_base, fundamentals)
        mu_momentum = apply_momentum_adjustment(mu_style, window_prices)
        mu_final = apply_mu_confidence_adjustment(mu_momentum, window_prices)

        try:
            weights, perf = optimize_portfolio(mu_final, sigma, risk_free_rate=0.015, regularize=True, gamma=0.1)
        except:
            continue

        results.append({
            'date': rebalance_date.strftime('%Y-%m-%d'),
            'exp_ret': perf[0],
            'vol': perf[1],
            'sharpe': perf[2],
            'top_holdings': dict(sorted(weights.items(), key=lambda x: -x[1])[:5])
        })

    df = pd.DataFrame(results)

    # 📤 儲存報告：JSON + Excel
    df.to_json("backtest_report.json", orient="records", indent=2)
    df.to_excel("backtest_report.xlsx", index=False)

    # 📊 產出圖表：Sharpe 時序圖
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(df['date'], df['sharpe'], marker='o')
    plt.title('Sharpe Ratio Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sharpe_trend.png")

    return df

