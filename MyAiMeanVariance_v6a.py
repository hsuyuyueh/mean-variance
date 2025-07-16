# 類別化改寫：AiMeanVariancePortfolio

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
from my_ai_module import gpt_contextual_rating
import matplotlib.pyplot as plt

nest_asyncio.apply()

# 建立 cache 資料夾
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class AiMeanVariancePortfolio:
    def __init__(self, tickers, prices, profile_level='P3'):  # 預設為成長型
        self.profile_level = profile_level  # P1, P2, P3, P4
        self.tickers = tickers
        self.prices = prices
        self.mu = None
        self.fundamentals = None
        self.rf_rate = 0.015
        self.capital = 10_000_000
        self.weights = None
        self.sigma = None
        self.mu_final = None

    def fetch_fundamentals(self):
        """
        抓取並快取基本面資料，每天只跑一次 yfinance
        cache 格式： cache/fundamentals_YYYYMMDD.json
        """
        today = datetime.today().strftime('%Y%m%d')
        cache_file = os.path.join(CACHE_DIR, f"fundamentals_{today}.json")
        if os.path.exists(cache_file):
            print(f"[快取] 載入基本面資料：{cache_file}")
            df = pd.read_json(cache_file)
            self.fundamentals = df
            return

        fundamentals = {}
        for tk in tqdm(self.tickers, desc="抓取基本面資料"):
            try:
                info = yf.Ticker(tk).info
                fundamentals[tk] = {
                    'pe': info.get('trailingPE', 20),
                    'roe': info.get('returnOnEquity', 0.1) * 100
                }
            except Exception:
                fundamentals[tk] = {'pe': 20, 'roe': 10}
        df = pd.DataFrame.from_dict(fundamentals, orient='index')
        df.to_json(cache_file, force_ascii=False, indent=2)
        print(f"[快取] 已儲存基本面資料至：{cache_file}")
        self.fundamentals = df

    def build_mu(self):
        mu_base = gpt_contextual_rating(self.tickers, horizon_months=3)
        pe_scaled = 1 / (self.fundamentals['pe'] + 5)
        roe_scaled = self.fundamentals['roe'] / 20
        adj_factor = 0.5 * pe_scaled + 0.5 * roe_scaled
        mu_style = mu_base * adj_factor
        mu_momentum = mu_style * (1 + 0.3 * ((self.prices.iloc[-1] / self.prices.iloc[-60]) - 1))
        vol = self.prices.pct_change().rolling(60).std().mean().fillna(0)
        scaled = 1 / (1 + 0.5 * vol)
        self.mu_final = mu_momentum * scaled

    def optimize(self):
        # 計算共變異數矩陣並最適化
        self.sigma = exp_cov(self.prices, span=180)
        ef = EfficientFrontier(self.mu_final, self.sigma, weight_bounds=(0, 1))
        ef.add_objective(L2_reg, gamma=0.1)
        ef.max_sharpe(risk_free_rate=self.rf_rate)
        self.weights = ef.clean_weights()
        self.performance = ef.portfolio_performance(risk_free_rate=self.rf_rate, verbose=False)

        # 計算 portfolio beta 並輸出結果
        beta = self.estimate_portfolio_beta()
        print("\n=== 最佳化結果（class 模式）===")
        print(f"→ 投組 β：{beta}")
        if self.profile_level == 'P1' and beta > 0.8:
            print("⚠️ 警告：此配置不適合保守型客戶 (C1)")
        elif self.profile_level == 'P2' and beta > 1.0:
            print("⚠️ 警告：此配置風險高於穩健型 (C2)")
        elif self.profile_level == 'P3' and beta > 1.3:
            print("⚠️ 警告：超過成長型 (C3) 可接受 β 區間")
        probability = norm.cdf(self.performance[2])
        for tk, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            if w > 0:
                try:
                    sector = yf.Ticker(tk).info.get('sector', 'N/A')
                except:
                    sector = 'N/A'
                print(f"{tk}: {w:.2%}, μ={self.mu_final.get(tk, 0):.2%}, Sector={sector}")
        print(f"\n預期年化報酬: {self.performance[0]:.2%}, 年化波動率: {self.performance[1]:.2%}, Sharpe: {self.performance[2]:.2f}, P: {probability:.2f}")

    def estimate_portfolio_beta(self):
        market = yf.download("^TWII", period="1y", auto_adjust=False)['Adj Close'].pct_change().dropna()
        betas = {}
        for tk in self.tickers:
            try:
                stock = yf.download(tk, period="1y", auto_adjust=False)['Adj Close'].pct_change().dropna()
                df = pd.concat([stock, market], axis=1).dropna()
                cov = df.cov().iloc[0, 1]
                var = df.iloc[:, 1].var()
                betas[tk] = cov / var
            except:
                betas[tk] = 1.0
        portfolio_beta = sum(self.weights.get(tk, 0) * betas.get(tk, 1.0) for tk in self.tickers)
        return round(portfolio_beta, 2)

    def save_weights(self, filepath="current_portfolio.json"):
        with open(filepath, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def backtest(self, rebalance_freq='2W'):
        results = []
        rebalance_days = pd.date_range(start=self.prices.index[0], end=self.prices.index[-1], freq=rebalance_freq)
        for date in rebalance_days:
            window_prices = self.prices[:date].dropna(axis=1, how='any')
            if len(window_prices) < 100:
                continue
            sigma = exp_cov(window_prices, span=180)
            mu_momentum = self.mu_final * (1 + 0.3 * ((window_prices.iloc[-1] / window_prices.iloc[-60]) - 1))
            vol = window_prices.pct_change().rolling(60).std().mean().fillna(0)
            mu_adjusted = mu_momentum * (1 / (1 + 0.5 * vol))
            try:
                ef = EfficientFrontier(mu_adjusted, sigma)
                ef.add_objective(L2_reg, gamma=0.1)
                ef.max_sharpe(risk_free_rate=self.rf_rate)
                w = ef.clean_weights()
                perf = ef.portfolio_performance(risk_free_rate=self.rf_rate)
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'exp_ret': perf[0],
                    'vol': perf[1],
                    'sharpe': perf[2],
                    'top_holdings': dict(sorted(w.items(), key=lambda x: -x[1])[:5])
                })
            except:
                continue
        df = pd.DataFrame(results)
        df.to_json("backtest_report.json", indent=2)
        df.to_excel("backtest_report.xlsx", index=False)
        plt.figure(figsize=(10,4))
        plt.plot(df['date'], df['sharpe'], marker='o')
        plt.title('Sharpe Ratio Over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("sharpe_trend.png")
        return df

# ===== 主程式使用範例 =====
if __name__ == '__main__':
    from fetch_0056_components import fetch_0056_components
    from fetch_0050_components import fetch_0050_components

    components = list({tk: name for tk, name in fetch_0050_components() + fetch_0056_components()}.items())
    tickers = [tk for tk, name in components if not tk.startswith("289")]
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    # 明確設定 auto_adjust=False，以保留 'Adj Close' 欄位
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

    model = AiMeanVariancePortfolio(tickers, prices)
    model.fetch_fundamentals()
    model.build_mu()
    model.optimize()
    model.save_weights()
    df_bt = model.backtest()
    print("\n回測平均 Sharpe:", df_bt['sharpe'].mean().round(2))
