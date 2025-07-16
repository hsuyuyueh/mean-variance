#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest_asyncio
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt

from fetch_0056_components import fetch_0056_components
from fetch_0050_components import fetch_0050_components
from my_ai_module import gpt_contextual_rating

nest_asyncio.apply()

# 輸出根目錄：使用當天日期
RUN_DATE = datetime.today().strftime('%Y%m%d')
OUTPUT_ROOT = os.path.join("./outputs", RUN_DATE)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

class AiMeanVariancePortfolio:
    def __init__(self, tickers, prices, profile_level='P3'):
        self.profile_level = profile_level
        self.tickers = tickers
        self.prices = prices
        self.mu_final = None
        self.fundamentals = None
        self.rf_rate = 0.015
        self.weights = None
        self.performance = None
        self.target_return = None

    def fetch_fundamentals(self):
        cache_file = os.path.join(OUTPUT_ROOT, f"fundamentals_{RUN_DATE}.json")
        if os.path.exists(cache_file):
            print(f"[快取] 載入基本面資料：{cache_file}")
            self.fundamentals = pd.read_json(cache_file)
            return
        fundamentals = {}
        for tk in tqdm(self.tickers, desc="抓取基本面資料"):
            try:
                info = yf.Ticker(tk).info
                fundamentals[tk] = {
                    'pe': info.get('trailingPE', 20),
                    'roe': info.get('returnOnEquity', 0.1) * 100
                }
            except:
                fundamentals[tk] = {'pe': 20, 'roe': 10}
        df = pd.DataFrame.from_dict(fundamentals, orient='index')
        df.to_json(cache_file, force_ascii=False, indent=2)
        print(f"[快取] 已儲存基本面資料至：{cache_file}")
        self.fundamentals = df

    def build_mu(self):
        cache_file = os.path.join(OUTPUT_ROOT, f"default_mu_cache_{RUN_DATE}.json")
        # my_ai_module 已做日期快取，可選擇讀寫此檔
        self.mu_final = gpt_contextual_rating(self.tickers, force=False)

    def optimize(self):
        sigma = exp_cov(self.prices, span=180)
        common = self.mu_final.index.intersection(sigma.index)
        mu = self.mu_final.loc[common]
        sigma = sigma.loc[common, common]
        ef1 = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
        ef1.max_sharpe(risk_free_rate=self.rf_rate)
        ret1, _, _ = ef1.portfolio_performance(risk_free_rate=self.rf_rate, verbose=False)
        self.target_return = min(ret1, mu.max() * 0.999)
        ef2 = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
        ef2.add_objective(L2_reg, gamma=0.1)
        ef2.efficient_return(target_return=self.target_return)
        self.weights = ef2.clean_weights()
        self.performance = ef2.portfolio_performance(risk_free_rate=self.rf_rate, verbose=False)

        beta = self.estimate_portfolio_beta()
        print("\n=== 最佳化結果 (class 模式) ===\n")
        print(f"→ 投組 β：{beta}")
        prob = norm.cdf(self.performance[2])
        for tk, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            if w > 0:
                sector = 'N/A'
                try:
                    sector = yf.Ticker(tk).info.get('sector', 'N/A')
                except:
                    pass
                print(f"{tk}: {w:.2%}, μ={mu.get(tk,0):.2%}, Sector={sector}")
        print(f"\n預期年化報酬: {self.performance[0]:.2%}, 年化波動率: {self.performance[1]:.2%}, Sharpe: {self.performance[2]:.2f}, P: {prob:.2f}\n")

    def estimate_portfolio_beta(self):
        market = yf.download("^TWII", period="1y", auto_adjust=False)['Adj Close'].pct_change().dropna()
        betas = {}
        for tk in self.tickers:
            try:
                stock = yf.download(tk, period="1y", auto_adjust=False)['Adj Close'].pct_change().dropna()
                df2 = pd.concat([stock, market], axis=1).dropna()
                cov = df2.cov().iloc[0,1]
                var = df2.iloc[:,1].var()
                betas[tk] = cov/var
            except:
                betas[tk] = 1.0
        return round(sum(self.weights.get(tk,0)*betas.get(tk,1) for tk in self.tickers), 2)

    def save_weights(self, filepath=None):
        path = filepath or os.path.join(OUTPUT_ROOT, "current_portfolio.json")
        with open(path, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def backtest(self, rebalance_freq='2W'):
        results = []
        dates = pd.date_range(self.prices.index[0], self.prices.index[-1], freq=rebalance_freq)
        for date in dates:
            window = self.prices[:date].dropna(axis=1, how='any')
            if len(window) < 100:
                continue
            sigma_bt = exp_cov(window, span=180)
            common_bt = self.mu_final.index.intersection(sigma_bt.index)
            mu_bt = self.mu_final.loc[common_bt]
            sigma_bt = sigma_bt.loc[common_bt, common_bt]
            mu_mom = mu_bt * (1 + 0.3 * ((window.iloc[-1]/window.iloc[-60]) - 1))
            vol = window.pct_change().rolling(60).std().mean().fillna(0)
            mu_adj = mu_mom * (1 / (1 + 0.5 * vol))
            try:
                ef_bt = EfficientFrontier(mu_adj, sigma_bt, weight_bounds=(0, 1))
                ef_bt.add_objective(L2_reg, gamma=0.1)
                ef_bt.efficient_return(target_return=self.target_return)
                weights_bt = ef_bt.clean_weights()
                perf_bt = ef_bt.portfolio_performance(risk_free_rate=self.rf_rate, verbose=False)
                top5 = dict(sorted(weights_bt.items(), key=lambda x: -x[1])[:5])
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'exp_ret': perf_bt[0],
                    'vol': perf_bt[1],
                    'sharpe': perf_bt[2],
                    'top_holdings': top5
                })
            except:
                continue
        df = pd.DataFrame(results)
        bt_json = os.path.join(OUTPUT_ROOT, "backtest_report.json")
        bt_xlsx = os.path.join(OUTPUT_ROOT, "backtest_report.xlsx")
        df.to_json(bt_json, indent=2)
        df.to_excel(bt_xlsx, index=False)
        plt.figure(figsize=(10, 4))
        plt.plot(df['date'], df['sharpe'], marker='o')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_ROOT, "sharpe_trend.png"))
        return df

if __name__ == '__main__':
    components = list({tk: name for tk, name in fetch_0050_components() + fetch_0056_components()}.items())
    tickers = [tk for tk, name in components if not tk.startswith("289")]
    # tickers = ['2330.TW', '2317.TW', '2454.TW']
    start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')
    prices = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']

    model = AiMeanVariancePortfolio(tickers, prices)
    model.fetch_fundamentals()
    model.build_mu()
    model.optimize()
    model.save_weights()
    df_bt = model.backtest()
    print("\n回測平均 Sharpe:", df_bt['sharpe'].mean().round(2))
