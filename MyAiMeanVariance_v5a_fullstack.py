#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest_asyncio
import pandas as pd
import yfinance as yf
from pypfopt.risk_models import sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg
from fetch_0056_components import fetch_0056_components
from fetch_0050_components import fetch_0050_components
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

    # 複合曝險調整因子
    market_factor = vix_factor * jnk_factor * drawdown_factor

    # 限制在 [0.4, 1.0] 之間
    return max(0.4, min(1.0, market_factor))
    vix = yf.download("^VIX", period="30d", interval="1d")['Close'].mean()
    jnk = yf.download("JNK", period="30d", interval="1d")['Close'].pct_change().mean()
    drawdown = recent_drawdown(prices, window=10)

    # 假設基準 VIX = 20, JNK 平穩 = 0.001, drawdown 上限 = 15%
    vix_factor = max(0.5, 1.0 - (vix - 20) / 40)
    jnk_factor = max(0.5, 1.0 + jnk * 10)
    drawdown_factor = 0.7 if drawdown > dd_limit else 1.0

    return vix_factor * jnk_factor * drawdown_factor

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
    max_dd_per_asset = drawdown.max()
    return max_dd_per_asset.max()

def apply_risk_controls(weights_raw, current_weights, prices,
                        cap=0.20, threshold=0.05, dd_limit=0.15, dd_leverage=0.8,
                        apply_market_risk=True):
    w_cap = {tk: min(w, cap) for tk, w in weights_raw.items()}
    w_adj = {}
    for tk, w in w_cap.items():
        current = current_weights.get(tk, 0)
        if abs(w - current) > threshold:
            w_adj[tk] = w
        else:
            w_adj[tk] = current
    if recent_drawdown(prices) > dd_limit:
        w_adj = {tk: w * dd_leverage for tk, w in w_adj.items()}

    # v4 強化：整體曝險比例動態調整
    if apply_market_risk:
        exposure_factor = compute_market_risk_factor(prices)
        w_adj = {tk: w * exposure_factor for tk, w in w_adj.items()}

    total = sum(w_adj.values())
    if total > 0:
        w_final = {tk: w / total for tk, w in w_adj.items()}
    else:
        w_final = w_adj
    return w_final

def load_current_portfolio():
    return {}

if __name__ == '__main__':
    components_0056 = fetch_0056_components()
    components_0050 = fetch_0050_components()
    combined_components = {code: name for code, name in components_0056 + components_0050}
    tickers = [tk for tk in combined_components.keys() if not tk.startswith("289")]  # 範例：排除純金融股

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    irx = yf.download("^IRX", period="7d", interval="1d")['Close'].mean() / 100
    rf_rate, capital = irx if irx else 0.015, 10_000_000

    prices = load_prices(tickers, start_date, end_date)
    sigma = sample_cov(prices)

    horizon_months = 3
    mu_base = gpt_contextual_rating(tickers, horizon_months=horizon_months)
    mu_momentum = apply_momentum_adjustment(mu_base, prices, window=60, weight=0.3)
    mu_final = apply_mu_confidence_adjustment(mu_momentum, prices, window=60, penalty=0.5)

    weights_raw, (exp_ret, vol, sharpe) = optimize_portfolio(
        mu_final, sigma, risk_free_rate=rf_rate, regularize=True, gamma=0.1)

    current_weights = load_current_portfolio()
    weights_final = apply_risk_controls(weights_raw, current_weights, prices)

    print(f"使用樣本期：{start_date} 到 {end_date}\n")
    print('=== 最佳化結果（Max Sharpe with L2 + Momentum μ + Confidence Filter + Risk Control）===')
    probability = norm.cdf(sharpe)
    for tk, w in weights_final.items():
        if w > 0:
            print(f'  {tk}: {w:.2%}')
    print(f"\n預期年化報酬: {exp_ret:.2%}, 年化波動率: {vol:.2%}, Sharpe: {sharpe:.2f}, Probability : {probability:.2f} \n")
    print(f'=== 資金配置 (總資金 {capital:,} TWD) ===')
    print(f'=== Top 持股解釋 (μ 與產業分類) ===')
    try:
        for tk, w in sorted(weights_final.items(), key=lambda x: -x[1])[:5]:
            mu_val = mu_final.get(tk, 0)
            info = yf.Ticker(tk).info
            sector = info.get('sector', 'N/A')
            print(f"  {tk}: {w:.2%}, μ={mu_val:.2%}, Sector={sector}")
    except Exception as e:
        print(f"  ⚠️ 無法取得個股說明: {e}")
    for tk, w in weights_final.items():
        amount = capital * w
        if amount > 0:
            print(f'  {tk}: {amount:,.0f} TWD')
