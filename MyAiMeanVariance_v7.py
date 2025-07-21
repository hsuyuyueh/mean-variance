#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest_asyncio
import pandas as pd
import yfinance as yf
import json
import os
import logging
import warnings

# 關閉所有 WARNING 與以下等級的 logging
logging.disable(logging.WARNING)
# 完全忽略任何 warnings
warnings.filterwarnings("ignore")

# 接著再去匯入 matplotlib
import matplotlib.pyplot as plt
import argparse, sys, json
from pathlib import Path
import shutil
import time

from datetime import datetime, timedelta
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg
from scipy.stats import norm
from tqdm import tqdm

from check_p_of_ticker import check_p_of_ticker
from fetch_0056_components import fetch_0056_components
from fetch_0050_components import fetch_0050_components
from fetch_00713_components import fetch_00713_components
from fetch_US_berkshire_components import fetch_US_berkshire_components
from fetch_US_harvard_components import fetch_US_harvard_components
from fetch_US_SPY_components import fetch_US_SPY_components
from my_ai_module import gpt_contextual_rating

import time, requests
from datetime import date


import collections
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import os
import re
import inspect

def lineno():
    """回傳呼叫此函式的行號"""
    return inspect.currentframe().f_back.f_lineno

# 範例
#print("這是行號：", lineno())

# ---- 1. 讀取 API Key ----
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TEJ_API_KEY           = os.getenv("TEJ_API_KEY")

# ---- 2. Rate‐limit 控制 ----
# Alpha Vantage: 5 calls/minute → 每 call 需 sleep 12s
def throttle_alpha_vantage():
    time.sleep(12)

# TEJ: trial 上限 500 calls/day, paid 2000/day → 用簡單的「日計數器」加上最小延遲
TEJ_DAILY_LIMIT = 500  # 若您已付費，請改為 2000
TEJ_COUNTER_FILE = "./cache/tej_count.json"

def throttle_tej():
    today = date.today().isoformat()
    # 載入計數器
    if os.path.exists(TEJ_COUNTER_FILE):
        with open(TEJ_COUNTER_FILE, "r") as f:
            cnt = json.load(f)
    else:
        cnt = {}
    used = cnt.get(today, 0)
    if used >= TEJ_DAILY_LIMIT:
        raise RuntimeError(f"今日 TEJ API 呼叫次數已達上限 ({TEJ_DAILY_LIMIT})")
    # 更新並儲存
    cnt[today] = used + 1
    with open(TEJ_COUNTER_FILE, "w") as f:
        json.dump(cnt, f)
    # 每次呼叫小延遲，避免 burst
    time.sleep(1)

def fetch_fundamentals_yahoo(ticker):
    """
    使用 yfinance 從 Yahoo Finance 取得基本面：PE、ROE，並自動計算營收年增率（若可用）。
    """
    tk = yf.Ticker(ticker)
    info = tk.info

    # 取得 PE、ROE
    pe = info.get("trailingPE") or info.get("forwardPE") or 0
    roe = info.get("returnOnEquity", 0) * 100

    # 營收年增率：嘗試從年度財報計算
    rev_growth = 0
    try:
        fin = tk.financials  # DataFrame，欄位為年度
        revenues = fin.loc["Total Revenue"]
        # 取最近兩年比較
        rev_growth = ((revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]) * 100
    except Exception as e:
        rev_growth = 0
        print(date, "line：", lineno(), e)

    return {
        "pe": float(pe),
        "roe": float(roe),
        "rev_growth": float(rev_growth)
    }
    
def fetch_fundamentals_tej(ticker):
    """
    從 TEJ API 取基本面：PE, ROE, 營收成長率等
    回傳 dict {'pe': float, 'roe': float, 'rev_growth': float, ...}
    """
    throttle_tej()
    url = f"https://api.tej.com.tw/v1/data/{ticker}/fundamental"
    headers = {"Authorization": f"Bearer {TEJ_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json().get("data", [{}])[0]
    return {
        'pe':         float(data.get("trailingPE", 0)),
        'roe':       float(data.get("returnOnEquity", 0)) * 100,
        'rev_growth': float(data.get("revenueGrowth", 0)) * 100
    }

def fetch_fundamentals_alpha_vantage(ticker):
    """
    從 Alpha Vantage 取基本面 (Company Overview endpoint)
    """
    throttle_alpha_vantage()
    url = (
      "https://www.alphavantage.co/query"
      f"?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    info = resp.json()
    return {
        'pe':         float(info.get("PERatio", 0)),
        'roe':       float(info.get("ReturnOnEquityTTM", 0)),
        'rev_growth': None  # AV Overview 無提供，或可另 call TIME_SERIES_CUSTOM
    }
    
# --- add this block (after existing imports) -----------------

def _detect_market(self):
    if all(tk.endswith('.TW') for tk in self.tickers):  return 'TW'
    if all(tk.endswith('.T')  for tk in self.tickers):  return 'JP'
    return 'US'

def _normalize_us_ticker(tk: str) -> str:
    """
    1. 刪掉 MoneyDJ 會附的 '.US'
    2. 把 BRK.B → BRK-B 這種類股別改成 Yahoo 用的 '-'
    """
    if tk.endswith(".US"):
        tk = tk[:-3]
    return tk.replace(".", "-")
# -------------------------------------------------------------

nest_asyncio.apply()

# ── 股價取得備援：Yahoo、Alpha Vantage + SQLite 快取 ──
import time, sqlite3, requests
from pandas_datareader import data as pdr

DB_PATH = "./cache/price_cache.sqlite3"
# 確保 cache 資料夾存在，避免無法開啟資料庫
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)  # 現在就能正常建立或開啟檔案了
conn.execute("""
CREATE TABLE IF NOT EXISTS price(
    symbol TEXT, date TEXT, adj_close REAL,
    PRIMARY KEY(symbol, date)
)
""")
conn.commit()

def fetch_price_yahoo(symbol, start, end):
    try:
        df = pdr.get_data_yahoo(symbol, start=start, end=end)
        return df["Adj Close"]
    except Exception as e:
        print(date, "line：", lineno(), e)
        return None

def fetch_price_alphaav(symbol, start, end, api_key, pause=12):
    # 先查 SQLite 快取
    cur = conn.execute(
        "SELECT date, adj_close FROM price WHERE symbol=? AND date BETWEEN ? AND ?",
        (symbol, start, end)
    )
    rows = cur.fetchall()
    if rows:
        s = pd.Series({r[0]: r[1] for r in rows})
        return s.sort_index()

    # 呼叫 Alpha Vantage
    resp = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": "full"
        },
        timeout=10
    )
    data = resp.json()
    ts = data.get("Time Series (Daily)")
    if not ts:
        print(f"[警告] Alpha Vantage 無日線資料 for {symbol}: {data.get('Note') or data.get('Error Message')}")
        return pd.Series(dtype=float)

    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)

    # 找出含 “adjusted close” 的欄位
    cols = [c for c in df.columns if "adjusted close" in c.lower()]
    if not cols:
        print(f"[警告] 找不到調整後收盤價欄位 for {symbol}，Available: {df.columns.tolist()}")
        return pd.Series(dtype=float)

    s = df[cols[0]].loc[start:end].astype(float)

    # 寫入快取
    for dt, val in s.items():
        conn.execute(
            "INSERT OR IGNORE INTO price(symbol,date,adj_close) VALUES(?,?,?)",
            (symbol, dt.strftime("%Y-%m-%d"), float(val))
        )
    conn.commit()
    time.sleep(pause)
    return s

def fetch_price(symbols, start, end, av_api_key=None):
    """一次下載所有 ticker 的價格，回傳 Adj Close DataFrame"""
    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    # 如果有多層欄位，取 Adj Close
    return data['Adj Close'] if 'Adj Close' in data else data


class AiMeanVariancePortfolio:
    def __init__(self, tickers, prices, market, OUTPUT_ROOT, RUN_DATE, profile_level='P3'):
        self.profile_level = profile_level
        self.tickers = tickers
        self.prices = prices
        self.market  = market          # NEW
        self.mu_final = None
        self.fundamentals = None
        self.rf_rate = 0.015
        self.weights = None
        self.performance = None
        self.target_return = None
        self.OUTPUT_ROOT = OUTPUT_ROOT
        self.RUN_DAT = RUN_DATE
        
        ## 1. 列出要預測的所有 tickers
        #print("Component tickers:", self.tickers)
        #
        ## 2. mu_final 來源：AI 資料夾裡有哪些 mu-*.txt
        #ai_dir = f"./outputs/20250721/TW/AI"
        #mu_files = [f for f in os.listdir(ai_dir) if f.startswith("mu-") and f.endswith(".txt")]
        #print("已產生的 μ 檔案：", [f.split('-')[1].split('.')[0] for f in mu_files])
        #
        ## 3. 價格資料：self.prices.columns
        #model._load_prices()  # 或直接檢查 model.prices
        #print("價格資料欄位：", list(model.prices.columns))
        #
        ## 找出缺少的 ticker
        #missing_mu = set(model.tickers) - set(f.split('-')[1].split('.')[0] for f in mu_files)
        #missing_price = set(model.tickers) - set(model.prices.columns)
        #print("缺少 μ 預測檔案的 tickers：", missing_mu)
        #print("缺少價格資料的 tickers：", missing_price)

    def fetch_fundamentals(self):
        cache_file = os.path.join(self.OUTPUT_ROOT, f"fundamentals_{self.RUN_DAT}.json")
        if os.path.exists(cache_file):
            print(f"[快取] 載入基本面資料：{cache_file}")
            self.fundamentals = pd.read_json(cache_file)
            return

        fundamentals = {}
        for tk in tqdm(self.tickers, desc="抓取基本面資料"):
            try:
                if self.market == 'TW':
                    info = fetch_fundamentals_yahoo(tk)
                else:
                    # JP 可套用 Alpha Vantage 或保留原 yfinance
                    info = fetch_fundamentals_yahoo(tk)
                fundamentals[tk] = info
            except Exception as e:
                print(date, "line：", lineno(), e)
                print(f"[警告] {tk} 基本面抓取失敗：{e}，使用預設值")
                fundamentals[tk] = {'pe':20, 'roe':10, 'rev_growth':0}
            time.sleep(0.2)

        df = pd.DataFrame.from_dict(fundamentals, orient='index')
        df.to_json(cache_file, force_ascii=False, indent=2)
        print(f"[快取] 已儲存基本面資料至：{cache_file}")
        self.fundamentals = df

    def build_mu(self):
        import os, json, pandas as pd, numpy as np

        # 1. 統一快取檔名（無日期後綴）
        cache_file = os.path.join(self.OUTPUT_ROOT, "default_mu_cache.json")

        # --- 先讀快取：若檔案存在就直接載入並回傳 ---
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as cf:
                mu_dict = json.load(cf)
            # 加上 market 後綴
            self.mu_final = pd.Series({
                f"{ticker}.{self.market}": float(val)
                for ticker, val in mu_dict.items()
            })
            print(f"[INFO] 已從快取讀取 μ：{cache_file}")
            return

        # 2. Step 1: 本地計算 historical μ
        log_ret = np.log(self.prices / self.prices.shift(1)).dropna()
        mu_local = log_ret.mean() * 252
        local_mu_path = os.path.join(self.OUTPUT_ROOT, f"local_mu_{self.RUN_DAT}.json")
        mu_local.to_json(local_mu_path, force_ascii=False, indent=2)
        print(f"[INFO] 已儲存本地 historical μ 至 {local_mu_path}")

        # 3. Step 2: 計算技術指標 (略，同您原本程式)
        # —— Step 2: 本地計算技術指標 —— 
        if self.prices.empty:
            tech_indicators = {
                "ma5":       {},
                "macd":      {},
                "kd_k":      {},
                "kd_d":      {},
                "year_line": {},
            }
        else:
            adj = self.prices
            ma5   = adj.rolling(window=5).mean().iloc[-1]
            ema12 = adj.ewm(span=12).mean()
            ema26 = adj.ewm(span=26).mean()
            macd  = (ema12 - ema26).iloc[-1]
            low9  = adj.rolling(9).min()
            high9 = adj.rolling(9).max()
            raw_k = (adj - low9) / (high9 - low9) * 100
            kd_k  = raw_k.iloc[-1]
            kd_d  = raw_k.rolling(window=3).mean().iloc[-1]
            year_line = adj.rolling(window=252).mean().iloc[-1]

            tech_indicators = {
                "ma5":       ma5.round(2).to_dict(),
                "macd":      macd.round(4).to_dict(),
                "kd_k":      kd_k.round(2).to_dict(),
                "kd_d":      kd_d.round(2).to_dict(),
                "year_line": year_line.round(2).to_dict(),
            }
            
        # 4. Step 3: 呼叫 AI
        self.mu_final = gpt_contextual_rating(
            tickers=self.tickers,
            base_mu=mu_local.to_dict(),
            tech_indicators=tech_indicators,
            force=False,
            OUTPUT_ROOT=self.OUTPUT_ROOT
        )

        # 5. Fallback: cache 與 AI txt 皆無，強行從 AI 資料夾讀取
        if self.mu_final.empty:
            print("[警告] cache 空檔，改從 AI 資料夾讀取 mu-*.txt")
            ai_folder = os.path.join(self.OUTPUT_ROOT, "AI")
            mu_dict = {}
            for fn in os.listdir(ai_folder):
                if fn.startswith("mu-") and fn.endswith(".txt"):
                    ticker = fn.split("-", 1)[1].split(".")[0] + "." + self.market
                    
                    
                with open(os.path.join(ai_folder, fn), "r", encoding="utf-8") as cf:
                    content = cf.read()
                    # 統一 JSON 擷取流程
                    pattern = r'(?s)^.*?(?=== RAW CONTENT ===)'
                    content = re.sub(pattern, '', content)
                    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
                    if json_match and json_match.group(1).strip():
                        json_str = json_match.group(1).strip()
                    else:
                        obj_match = re.search(r"(\{[\s\S]*?\})", content)
                        if obj_match:
                            json_str = obj_match.group(1)
                        else:
                            raise ValueError("無法從 AI 回應中擷取到 JSON 物件")
                    print(f"\n\n\n[快取] {json_str}")
                    data = json.loads(json_str)
                    mu_raw = float(data.get("mu_prediction", 0.0))
                    mu_dict[ticker] = mu_raw / 100 if mu_raw > 1 else mu_raw
                    continue
                    

            self.mu_final = pd.Series(mu_dict)

        # 6. 最後寫入統一快取
        with open(cache_file, "w", encoding="utf-8") as cf:
            json.dump({k.split(f".{self.market}")[0]: v
                       for k, v in self.mu_final.to_dict().items()},
                      cf, indent=2)
        print(f"[INFO] 已寫入 μ 快取：{cache_file}")

    def optimize(self):
        sigma = exp_cov(self.prices, span=180)
        # 加上一點 jitter，確保 covariance matrix 正定
        import numpy as np
        sigma += np.eye(sigma.shape[0]) * 1e-4

        # ——— 以 self.tickers 為基準，篩選同時存在於 mu_final、covariance、與原始 ticker list ———
        candidates = [tk for tk in self.tickers
                      if tk in self.mu_final.index
                      and tk in sigma.index]

        if not candidates:
            print("[錯誤] 無法進行最佳化：mu 與 cov 的交集為空。")
            print(f"  • 原始 tickers: {self.tickers}")
            print(f"  • mu_final.index: {list(self.mu_final.index)}")
            print(f"  • covariance.index: {list(sigma.index)}")
            print("line：", lineno())
            return

        # 只取有共通的 ticker 進行最佳化
        mu = self.mu_final.loc[candidates]
        sigma = sigma.loc[candidates, candidates]
        
        ef1 = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
        # 嘗試以使用者設定的無風險利率計算最大化 Sharpe 組合
        try:
            ef1.max_sharpe(risk_free_rate=self.rf_rate)
        except ValueError as e:
            # 當所有資產預期報酬都 ≤ 無風險利率時，改用無風險利率=0 重算
            print(f"[警告] {e}，改用無風險利率=0 重新計算 max_sharpe")
            print(date, "line：", lineno(), e)
            ef1.max_sharpe(risk_free_rate=0)
        # 取得該組合的預期年化報酬
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
                except Exception as e:
                    print(date, "line：", lineno(), e)
                    pass
                print(f"{tk}: {w:.2%}, μ={mu.get(tk,0):.2%}, Sector={sector}")
                args.source = tk
                args.sep='\t'
                args.period='6mo'
                args.window=63
                
                args.Increase=1+(self.performance[1]/2)
                args.T=21
                check_p_of_ticker(args)
                                
                args.Increase=1+(self.performance[1]/2)
                args.T=10
                check_p_of_ticker(args)
                                
                args.Increase=1+(self.performance[1]/2)
                args.T=5
                check_p_of_ticker(args)
                
                args.Increase=1-self.performance[1]
                args.T=63
                check_p_of_ticker(args)
                
                args.Increase=1-(self.performance[1]/2)
                args.T=21
                check_p_of_ticker(args)
                
                args.Increase=1-(self.performance[1]/2)
                args.T=10
                check_p_of_ticker(args)
                
                args.Increase=1-(self.performance[1]/2)
                args.T=5
                check_p_of_ticker(args)
                
                args.Increase=1-(self.performance[1]/2)
                args.T=2
                check_p_of_ticker(args)
                
                args.Increase=1+self.performance[1]
                args.T=63
                check_p_of_ticker(args)

        print(f"\n預期年化報酬: {self.performance[0]:.2%}, 年化波動率: {self.performance[1]:.2%}, Sharpe: {self.performance[2]:.2f}, P: {prob:.2f}\n")

    def estimate_portfolio_beta(self):
        idx_map = {'TW':'^TWII', 'US':'^GSPC', 'JP':'^N225'}
        # 1. 先下載基準指數報酬
        bm_df = yf.download(idx_map[self.market], period="1y", progress=False, auto_adjust=False)
        if "Adj Close" in bm_df.columns:
            benchmark = bm_df["Adj Close"].pct_change().dropna()
        else:
            benchmark = bm_df["Close"].pct_change().dropna()

        # 2. 針對每個股票計算 beta
        betas = {}
        for tk in self.tickers:
            try:
                stk_df = yf.download(tk, period="1y", progress=False, auto_adjust=False)
                if "Adj Close" in stk_df.columns:
                    stk = stk_df["Adj Close"].pct_change().dropna()
                else:
                    stk = stk_df["Close"].pct_change().dropna()
                df2 = pd.concat([stk, benchmark], axis=1).dropna()
                betas[tk] = df2.cov().iloc[0,1] / df2.iloc[:,1].var()
            except Exception as e:
                print(date, "line：", lineno(), e)
                betas[tk] = 1.0

        # 3. 加權平均所有持股的 beta
        return round(sum(self.weights.get(tk, 0) * betas.get(tk, 1.0)
                         for tk in self.tickers), 2)

    def save_weights(self, filepath=None):
        path = filepath or os.path.join(OUTPUT_ROOT, "current_portfolio.json")
        with open(path, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def fetch_benchmark_series(self):
        import yfinance as yf
        import pandas as pd

        # 下載基準資料，確保同時回傳 Close 與 Adj Close
        if self.market == 'TW':
            df_bench = yf.download("0050.TW", period="1y", progress=False, auto_adjust=False)
        else:  # US
            df_bench = yf.download("^GSPC", period="1y", progress=False, auto_adjust=False)

        # 優先使用 ‘Adj Close’，若無此欄位則退而求其次使用 ‘Close’
        if 'Adj Close' in df_bench.columns:
            bm = df_bench['Adj Close']
        else:
            bm = df_bench['Close']

        return bm.dropna()
    
    def plot_top_holdings_summary(self, df):
        holding_counter = collections.Counter()
        for top_dict in df['top_holdings']:
            holding_counter.update(top_dict)

        top_items = holding_counter.most_common(10)
        tickers = [k for k, _ in top_items]
        counts = [v for _, v in top_items]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(tickers, counts)
        plt.xlabel("Top Holdings (Most Frequent)")
        plt.ylabel("Appearance Count")
        plt.title("Top Holdings Frequency During Backtest")
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        output_path = os.path.join(self.OUTPUT_ROOT, "top_holdings_chart.png")
        plt.savefig(output_path)
        print(f"[資訊] 已儲存 top holdings 圖表：{output_path}")

    def plot_sharpe_beta_map(self, df):
        if 'sharpe' in df.columns and 'beta' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='beta', y='sharpe', hue='market_cond', palette='Set2', s=80)
            plt.axhline(0.3, ls='--', c='gray', label='Sharpe Threshold 0.3')
            plt.axvline(1.0, ls='--', c='gray', label='Beta=1')
            plt.xlabel('Portfolio Beta')
            plt.ylabel('Sharpe Ratio')
            plt.title('Sharpe vs Beta by Market Condition')
            plt.legend()
            plt.tight_layout()
            output_path = os.path.join(self.OUTPUT_ROOT, "sharpe_vs_beta_map.png")
            plt.savefig(output_path)
            print(f"[資訊] 已儲存 Sharpe vs Beta 圖表：{output_path}")

    def plot_sharpe_winrate_map(self, df):
        if 'sharpe' in df.columns and 'market_cond' in df.columns:
            df_copy = df.copy()
            df_copy['win'] = df_copy['exp_ret'] > df_copy['benchmark_value'].pct_change(periods=1).shift(-1) * 252
            df_copy['win'] = df_copy['win'].astype(int)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_copy, x='sharpe', y='win', hue='market_cond', palette='coolwarm', s=80)
            plt.xlabel('Sharpe Ratio')
            plt.ylabel('Victory (1=beat benchmark)')
            plt.title('Sharpe vs Victory by Market Condition')
            plt.yticks([0, 1])
            plt.tight_layout()
            output_path = os.path.join(self.OUTPUT_ROOT, "sharpe_vs_victory_map.png")
            plt.savefig(output_path)
            print(f"[資訊] 已儲存 Sharpe vs Victory 圖表：{output_path}")

    def _label_market_condition(self, date, vix_series, sp500_series):
        if date not in vix_series.index or date not in sp500_series.index:
            return "unknown"
        vix = vix_series.loc[date]
        sp = sp500_series.loc[date]
        ma200 = sp500_series.rolling(200).mean().loc[date]
        if vix < 15 and sp > ma200:
            return "bull"
        elif vix > 25 and sp < ma200:
            return "bear"
        else:
            return "neutral"

    def generate_html_report(self):
        html_path = Path(self.OUTPUT_ROOT) / "backtest_summary.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("""
            <html><head><meta charset='utf-8'><title>Backtest Summary</title></head><body>
            <h1>📊 Backtest Summary</h1>
            <ul>
              <li><img src='top_holdings_chart.png' width='600'></li>
              <li><img src='sharpe_vs_beta_map.png' width='600'></li>
              <li><img src='sharpe_vs_victory_map.png' width='600'></li>
              <li><img src='portfolio_vs_benchmark.png' width='600'></li>
            </ul>
            <p>For full data, see <code>backtest_report.xlsx</code> and <code>backtest_summary.txt</code>.</p>
            </body></html>
            """)
        print(f"[HTML] 已產出 HTML 報告：{html_path}")

    def plot_tracking_error(self, df):
        if 'mu_prediction' in df.columns and 'realized_return' in df.columns:
            df['tracking_error'] = df['mu_prediction'] - df['realized_return']
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='mu_prediction', y='realized_return', data=df)
            plt.xlabel('μ Prediction')
            plt.ylabel('Realized Return')
            plt.title('μ Prediction vs Realized Return')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.OUTPUT_ROOT, "mu_vs_realized_return.png"))

            plt.figure(figsize=(8, 4))
            sns.histplot(df['tracking_error'], bins=20, kde=True)
            plt.title('Tracking Error Distribution')
            plt.xlabel('μ - Realized Return')
            plt.tight_layout()
            plt.savefig(os.path.join(self.OUTPUT_ROOT, "tracking_error_distribution.png"))

            # summary
            avg_error = df['tracking_error'].mean()
            rmse = np.sqrt(np.mean(df['tracking_error']**2))
            summary_path = os.path.join(self.OUTPUT_ROOT, f"backtest_summary.txt")
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(f"\nTracking Error 統計：\n")
                f.write(f"平均誤差（μ - 實際）：{avg_error:.2%}\n")
                f.write(f"均方根誤差（RMSE）： {rmse:.2%}\n")


    def plot_weight_change(self, weight_list):
        diffs = []
        for i in range(1, len(weight_list)):
            w1 = weight_list[i-1]
            w2 = weight_list[i]
            tickers = list(set(w1) | set(w2))
            delta = sum(abs(w1.get(tk, 0) - w2.get(tk, 0)) for tk in tickers)
            diffs.append(delta)
        plt.figure(figsize=(10, 4))
        plt.plot(diffs, marker='o')
        plt.title('Portfolio Weight Change Over Time')
        plt.xlabel('Rebalance Step')
        plt.ylabel('Weight Δ (L1 norm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUTPUT_ROOT, "weight_change_trend.png"))
        print("[資訊] 已儲存權重變動圖：weight_change_trend.png")

    def plot_herfindahl_index(self, df):
        hhi_list = []
        for top_dict in df['top_holdings']:
            weights = np.array(list(top_dict.values()))
            hhi = np.sum(weights**2)
            hhi_list.append(hhi)
        plt.figure(figsize=(10, 4))
        plt.plot(hhi_list, marker='o')
        plt.title('Herfindahl Index (Concentration) Over Time')
        plt.xlabel('Rebalance Step')
        plt.ylabel('HHI')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUTPUT_ROOT, "herfindahl_index_trend.png"))
        print("[資訊] 已儲存持股集中度圖：herfindahl_index_trend.png")

    def plot_stock_hit_rate(self, df):
        from collections import defaultdict
        hit_counter = defaultdict(lambda: [0, 0])
        for row in df.itertuples():
            for tk, mu in row.top_holdings.items():
                if tk in self.prices.columns:
                    p_start = self.prices[tk].loc[:row.date].iloc[-1]
                    future_window = self.prices[tk].loc[row.date:].head(5)
                    if len(future_window) >= 2:
                        p_end = future_window.iloc[-1]
                        ret = (p_end / p_start - 1) * 252
                        hit = np.sign(mu) == np.sign(ret)
                        hit_counter[tk][0] += int(hit)
                        hit_counter[tk][1] += 1
        tk_list = [k for k, v in hit_counter.items() if v[1] >= 3]
        acc_list = [hit_counter[k][0]/hit_counter[k][1] for k in tk_list]
        plt.figure(figsize=(10, 4))
        plt.bar(tk_list, acc_list)
        plt.ylabel("Hit Rate")
        plt.title("Per-stock μ Direction Accuracy")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUTPUT_ROOT, "stock_hit_rate.png"))
        print("[資訊] 已儲存個股方向命中率圖：stock_hit_rate.png")

    def generate_diagnostic_summary(self, df):
        summary_path = os.path.join(self.OUTPUT_ROOT, "diagnostic_report.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("📊 Diagnostic Report for Portfolio Strategy\n")
            f.write("========================================\n\n")
            # === Tracking Error Auto Risk Rule ===
            if 'tracking_error' in df.columns:
                avg_error = df['tracking_error'].mean()
                rmse = np.sqrt(np.mean(df['tracking_error']**2))
                f.write(f"📌 μ 預測追蹤誤差：\n  • 平均誤差：{avg_error:.2%}\n  • 均方根誤差（RMSE）：{rmse:.2%}\n\n")
                if rmse > 0.04:
                    self.target_return = max(self.rf_rate + 0.005, self.target_return * 0.8)
                    self.risk_control_note = f"⚠️ 啟動風控：Tracking Error 過高（RMSE = {rmse:.2%}），已下調目標報酬至 {self.target_return:.2%}\n"
                    f.write(self.risk_control_note + "\n")
            # === Sharpe Ratio Auto Risk Rule ===
            if 'sharpe' in df.columns and 'beta' in df.columns:
                avg_sharpe = df['sharpe'].mean()
                avg_beta = df['beta'].mean()
                f.write(f"📌 報酬風險指標：\n  • 平均 Sharpe Ratio：{avg_sharpe:.2f}\n  • 平均 Beta 值：{avg_beta:.2f}\n\n")
                if avg_sharpe < 0.2:
                    self.target_return = max(self.rf_rate + 0.002, self.target_return * 0.85)
                    self.risk_control_note2 = f"⚠️ 啟動風控：Sharpe Ratio 偏低（平均 {avg_sharpe:.2f}），已下調目標報酬至 {self.target_return:.2%}\n"
                    f.write(self.risk_control_note2 + "\n")
                # === Beta Limit Auto Risk Rule ===
                if avg_beta > 1.3:
                    self.max_beta_limit = 1.0
                    self.risk_control_note3 = f"⚠️ 啟動風控：Beta 過高（平均 {avg_beta:.2f}），建議調整最佳化目標以限制波動風險\n"
                    f.write(self.risk_control_note3 + "\n")
            # === HHI Smooth Auto Rule ===
            if 'top_holdings' in df.columns:
                hhi_list = [sum(np.square(list(d.values()))) for d in df['top_holdings']]
                avg_hhi = np.mean(hhi_list)
                f.write(f"📌 組合集中度：\n  • 平均 Herfindahl 指數（HHI）：{avg_hhi:.3f}\n\n")
                if avg_hhi > 0.4:
                    self.hhi_smoothing_enabled = True
                    self.risk_control_note4 = f"⚠️ 啟動風控：組合集中度過高（HHI = {avg_hhi:.3f}），建議使用權重平滑化處理\n"
                    f.write(self.risk_control_note4 + "\n")
            if hasattr(self, 'plot_weight_change'):
                f.write(f"📌 權重穩定性：\n  • 已生成變動圖檔，可視覺化調倉幅度\n\n")
            f.write("🔍 建議檢查指標：\n")
            f.write("  - 若 Tracking Error 高，μ 預測可能需修正\n")
            f.write("  - 若 HHI > 0.4，組合可能過度集中\n")
            f.write("  - 若 Sharpe 高但 Beta 也高，代表風險來自高曝險\n")
            f.write("  - 若頻繁調倉，考慮加強正則化或改進預測穩定性\n")
            f.write("\n✅ 本報告已根據最新回測結果產生\n")
        print(f"[資訊] 已產出診斷報告：{summary_path}")

    def fetch_vix_benchmark(self):
        import yfinance as yf
        import pandas as pd

        # 取得 VIX 與 Benchmark（TW:0050, US:^GSPC）
        if self.market == 'TW':
            df_vix   = yf.download("^VIX",    period="1y", progress=False, auto_adjust=False)
            df_bench = yf.download("0050.TW",  period="1y", progress=False, auto_adjust=False)
        else:
            df_vix   = yf.download("^VIX",   period="1y", progress=False, auto_adjust=False)
            df_bench = yf.download("^GSPC",  period="1y", progress=False, auto_adjust=False)

        vix   = df_vix['Adj Close'] if 'Adj Close' in df_vix.columns else df_vix['Close']
        bench = df_bench['Adj Close'] if 'Adj Close' in df_bench.columns else df_bench['Close']
        return vix.dropna(), bench.dropna()

    def classify_market(vix_value, bench_ret):
        if vix_value > 25 or bench_ret < -0.05:
            return "bear"
        elif vix_value < 15 and bench_ret > 0.05:
            return "bull"
        else:
            return "neutral"
            
## backtest(self, rebalance_freq='2W', window_length=180):
## 📌 一、μ 預測品質（預測能力）
## plot_tracking_error()：比較 AI 預測與實際報酬的偏差與穩定性
## ⟶ 可以評估模型是否過度樂觀／悲觀、是否準確預測報酬趨勢
## 
## plot_stock_hit_rate()：每檔個股 μ 方向命中率
## ⟶ 檢查哪些股票容易過度預測錯誤，可考慮「排除持股」或加權調整
## 
## 📌 二、風險分散程度（集中度風險）
## plot_herfindahl_index()：持股集中度指標（HHI）
## ⟶ 若 HHI > 0.4 表示組合可能「單壓」某幾檔，違反分散原則
## 
## 📌 三、策略穩定性與可解性
## plot_weight_change()：每期權重變動程度（L1 norm）
## ⟶ 若頻繁劇變，可能 μ 資訊不穩定，或模型在特定市場情況過於敏感
## 
## 📌 四、α 與風險的關聯
## plot_sharpe_beta_map()：Sharpe vs Beta 散點圖
## ⟶ 評估投資績效是否來自「高風險換高報酬」或「有效資訊壓制風險」
## 
## plot_sharpe_winrate_map()：Sharpe vs 勝率圖
## ⟶ 若 Sharpe 高但勝率低，可能存在不對稱損益結構（賭一擊）
## 
## 📌 五、最常出現持股的角色
## plot_top_holdings_summary()
## ⟶ 哪些股票是常出現主力？可能是 AI 特別偏好／特別準的標的
## 
## 📌 六、結果的視覺報告與決策基礎
## HTML 報告整合：讓回測結果可傳遞給非技術利害人使用
## ⟶ 支援「資料驅動決策」與策略可溝通性
## 
## 📌 七、模型是否過度複雜化（風控角度）
## Tracking error + weight change + HHI 同時偏高
## ⟶ 表示 μ 太激進，模型輸出不穩定，應降階或正則化更強

    def backtest(self, rebalance_freq='2W', window_length=180):
        results = []
        portfolio_value = [1_000_000]
        dates_bt = []
        benchmark_value = [1_000_000]

        if self.prices.empty:
            print("[錯誤] 無法回測：價格資料為空，請確認是否成功取得股價資料。")
            return pd.DataFrame()

        vix_series, sp500_series = self.fetch_vix_benchmark()
        benchmark_series = self.fetch_benchmark_series()
        #dates = pd.date_range(self.prices.index[0], self.prices.index[-1], freq=rebalance_freq)
        # 如果你用 '1M' 會出現 FutureWarning，可改用 'ME' (month end)
        freq = rebalance_freq.replace("1M", "ME")
        dates = pd.date_range(self.prices.index[0], self.prices.index[-1], freq=freq)

        win_count = 0
        excess_returns = []

        for date in dates:
            window = self.prices.loc[:date].tail(window_length).dropna(axis=1, how='any')
            # 只在拿到完整 window_length 天數的資料時才做回測
            if len(window) < window_length:
                print("line：", lineno())
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

                use_target = self.target_return
                # PyPortfolioOpt 新版已無 expected_sharpe()，先嘗試呼叫，若不存在則跳過
                try:
                    sharpe_est = ef_bt.expected_sharpe()
                except AttributeError as e:
                    sharpe_est = None
                    print(date, "line：", lineno(), e)
                if sharpe_est is not None and sharpe_est < 0.3:
                    use_target = self.rf_rate + 0.01
                    ef_bt.add_objective(L2_reg, gamma=0.3)

                #ef_bt.efficient_return(target_return=use_target)
                use_target = self.target_return
                # 若 use_target 超過 mu_adj 的最大值，則自動調整為最大可行值的 99.9%
                max_ret = mu_adj.max()
                if use_target >= max_ret:
                    print(f"[警告] target_return {use_target:.4f} ≥ 最大可行報酬 {max_ret:.4f}，已自動調整")
                    use_target = max_ret * 0.999
                ef_bt.efficient_return(target_return=use_target)
                
                weights_bt = ef_bt.clean_weights()
                perf_bt = ef_bt.portfolio_performance(risk_free_rate=self.rf_rate, verbose=False)
                top5 = dict(sorted(weights_bt.items(), key=lambda x: -x[1])[:5])

                # === 計算該期 beta ===
                idx_map = {'TW':'^TWII', 'US':'^GSPC', 'JP':'^N225'}
                #bm_ret = yf.download(idx_map[self.market], period="1y", progress=False)["Adj Close"].pct_change().dropna()
                bm_df = yf.download(idx_map[self.market], period="1y", progress=False, auto_adjust=False)
                if "Adj Close" in bm_df.columns:
                    bm_ret = bm_df["Adj Close"].pct_change().dropna()
                else:
                    bm_ret = bm_df["Close"].pct_change().dropna()
                betas = {}
                for tk in weights_bt:
                    try:
                        stk_ret = yf.download(tk, period="1y", progress=False)["Adj Close"].pct_change().dropna()
                        dfb = pd.concat([stk_ret, bm_ret], axis=1).dropna()
                        beta_val = dfb.cov().iloc[0,1] / dfb.iloc[:,1].var()
                        betas[tk] = beta_val
                    except Exception as e:
                        print(date, "line：", lineno(), e)
                        betas[tk] = 1.0
                portfolio_beta = round(sum(weights_bt.get(tk, 0) * betas.get(tk, 1.0) for tk in weights_bt), 3)

                Δt = 14 / 252
                last_value = portfolio_value[-1]
                new_value = last_value * (1 + perf_bt[0] * Δt)
                portfolio_value.append(new_value)
                dates_bt.append(date)

                if date in benchmark_series.index and date - pd.Timedelta(days=14) in benchmark_series.index:
                    p0 = benchmark_series.loc[date - pd.Timedelta(days=14)]
                    p1 = benchmark_series.loc[date]
                    bm_ret_val = (p1 / p0) - 1
                    benchmark_value.append(benchmark_value[-1] * (1 + bm_ret_val))
                    excess_returns.append(perf_bt[0] - bm_ret_val * 252)
                    if perf_bt[0] > bm_ret_val * 252:
                        win_count += 1
                else:
                    benchmark_value.append(benchmark_value[-1])

                condition = self._label_market_condition(date, vix_series, sp500_series)

                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'exp_ret': perf_bt[0],
                    'vol': perf_bt[1],
                    'sharpe': perf_bt[2],
                    'top_holdings': top5,
                    'market_cond': condition,
                    'beta': portfolio_beta
                })

            except Exception as e:
                print(date, "line：", lineno(), e)
                continue

        df = pd.DataFrame(results)
        if 'market_cond' not in df.columns:
            df['market_cond'] = 'unknown'
        # ——— 新增：若回測結果為空，跳過後續繪圖與報告生成 ———
        if df.empty:
            print("[資訊] 回測結果為空，無法繪製圖表或生成報告，已跳過。")
            print("line：", lineno())
            return df
        if len(portfolio_value) > 1:
            value_series = pd.Series(portfolio_value[1:], index=dates_bt)
            benchmark_series_sim = pd.Series(benchmark_value[1:], index=dates_bt)
            rolling_max = value_series.cummax()
            drawdowns = (value_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            # 用 index 對齊 DataFrame，再賦值
            df['portfolio_value'] = value_series.reindex(df.index)
            df['benchmark_value'] = benchmark_series_sim.reindex(df.index)
            df['drawdown']        = drawdowns.reindex(df.index)
        else:
            max_drawdown = None

        #avg_excess = sum(excess_returns) / len(excess_returns) if excess_returns else 0
        # 计算平均超额报酬，并确保为标量（不是 Series）
        raw_excess = sum(excess_returns) / len(excess_returns) if excess_returns else 0
        try:
            # 如果 accidental 变成了 Series，就取第一个元素
            avg_excess = float(raw_excess)
        except Exception:
            avg_excess = raw_excess.iloc[0] if hasattr(raw_excess, 'iloc') else raw_excess
        win_rate = win_count / max(len(df),1)

        bt_json = os.path.join(self.OUTPUT_ROOT, "backtest_report.json")
        bt_xlsx = os.path.join(self.OUTPUT_ROOT, "backtest_report.xlsx")
        df.to_json(bt_json, indent=2)
        df.to_excel(bt_xlsx, index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(df['date'], df['portfolio_value'], label='Portfolio')
        plt.plot(df['date'], df['benchmark_value'], label='Benchmark')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.OUTPUT_ROOT, "portfolio_vs_benchmark.png"))

        summary_path = os.path.join(self.OUTPUT_ROOT, f"backtest_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"最大跌幅 Max Drawdown: {max_drawdown:.2%}\n")
            f.write(f"胜率（打败基准）: {win_count} / {len(df)} = {win_rate:.2%}\n")
            # 这里 avg_excess 已经是 float，可以安全使用百分比格式
            f.write(f"平均超额报酬: {avg_excess:.2%}\n")

        # plot top holdings chart
        self.plot_top_holdings_summary(df)
        # plot sharpe vs beta
        self.plot_sharpe_beta_map(df)
        # plot sharpe vs victory
        self.plot_sharpe_winrate_map(df)
        # plot tracking error
        self.plot_tracking_error(df)
        # plot weight change
        self.plot_weight_change([row['top_holdings'] for _, row in df.iterrows()])
        # plot herfindahl index
        self.plot_herfindahl_index(df)
        # plot stock μ hit rate
        self.plot_stock_hit_rate(df)
        # generate HTML summary
        self.generate_html_report()
        # generate diagnostic report
        self.generate_diagnostic_summary(df)
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', default='TW', choices=['TW', 'US', 'JP'],
                        help='TW=台股、US=美股、JP=日股')
    
    args = parser.parse_args(sys.argv[1:])
    # 輸出根目錄：使用當天日期
    RUN_DATE = datetime.today().strftime('%Y%m%d')
    OUTPUT_ROOT = os.path.join("./outputs", RUN_DATE)
    OUTPUT_ROOT = os.path.join(OUTPUT_ROOT, args.market)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.environ['OUTPUT_ROOT'] = OUTPUT_ROOT
    DB_PATH = "./cache/price_cache.sqlite3"
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    MARKET_SOURCES = {
        "TW": lambda: [tk for tk, _ in {**dict(fetch_0050_components(OUTPUT_ROOT)),
                                        **dict(fetch_00713_components(OUTPUT_ROOT)),
                                        **dict(fetch_0056_components(OUTPUT_ROOT))}.items()
                       if not tk.startswith("289")],
        "US": lambda: [tk for tk, _ in {**dict(fetch_US_harvard_components(OUTPUT_ROOT)), **dict(fetch_US_berkshire_components(OUTPUT_ROOT))}.items()],
        "JP": lambda: json.load(open("manual_jp_list.json"))  # e.g. ["7203.T","9984.T",...]
                           if os.path.exists("manual_jp_list.json") else ["7203.T","9984.T"]
    }
    tickers = MARKET_SOURCES[args.market]()
    #tickers = ['2330.TW', '2317.TW', '2454.TW']
    #tickers = ['2330.TW', '2317.TW']
    if args.market == "US":                       # 只清美股
        tickers = sorted({ _normalize_us_ticker(t) for t in tickers })
    start   = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end     = datetime.today().strftime('%Y-%m-%d')
    
    # 改用備援方案取得價格
    if args.market == "TW":
        prices = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    if args.market == "US": 
        prices = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
        #prices = fetch_price(
        #    tickers,
        #    start,
        #    end,
        #    av_api_key=ALPHA_VANTAGE_API_KEY
        #).dropna(axis=1, how="all")

    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"[警告] 無法取得以下股價：{sorted(missing)}")

    
    rf_table = {'TW':0.015, 'US':0.045, 'JP':0.002}
    

    
    model   = AiMeanVariancePortfolio(
                 tickers, prices,
                 market=args.market,              # 傳進去
                 OUTPUT_ROOT=OUTPUT_ROOT,
                 RUN_DATE=RUN_DATE,
                 profile_level='P3'
              )
    model.rf_rate = rf_table[args.market]         # 依市場覆寫無風險利率
    model.fetch_fundamentals()
    model.build_mu()
    # —— 新增 Debug：列出缺少的 μ 預測與價格資料 —— 
    missing_mu = set(model.tickers) - set(model.mu_final.index)
    missing_price = set(model.tickers) - set(model.prices.columns)

    print("=== Debug: μ 與價格資料檢查 ===")
    print("原始 tickers         :", model.tickers)
    print("已載入 mu_final.index:", list(model.mu_final.index))
    print("已載入 price columns :", list(model.prices.columns))
    print("缺少 μ 預測的 tickers:", sorted(missing_mu))
    print("缺少價格資料的 tickers:", sorted(missing_price))
    print("=== End Debug ===")
    model.optimize()
    model.save_weights()
    rebalance_freq="2W"
    print("line：", lineno())
    df_bt = model.backtest(rebalance_freq=rebalance_freq)
    # 保護性檢查：確保有回測結果且包含 sharpe 欄位才列印，否則提示錯誤
    if not df_bt.empty and 'sharpe' in df_bt.columns:
        print("\n{rebalance_freq}回測平均 Sharpe:", df_bt['sharpe'].mean().round(2))
    else:
        print("\n[錯誤] 無法計算 Sharpe：回測結果為空或缺少 'sharpe' 欄位。")
    print("line：", lineno())
    rebalance_freq="1W"
    df_bt = model.backtest(rebalance_freq=rebalance_freq)
    # 保護性檢查：確保有回測結果且包含 sharpe 欄位才列印，否則提示錯誤
    if not df_bt.empty and 'sharpe' in df_bt.columns:
        print("\n{rebalance_freq}回測平均 Sharpe:", df_bt['sharpe'].mean().round(2))
    else:
        print("\n[錯誤] 無法計算 Sharpe：回測結果為空或缺少 'sharpe' 欄位。")
    print("line：", lineno())
    rebalance_freq="1M"
    df_bt = model.backtest(rebalance_freq=rebalance_freq)
    # 保護性檢查：確保有回測結果且包含 sharpe 欄位才列印，否則提示錯誤
    if not df_bt.empty and 'sharpe' in df_bt.columns:
        print("\n{rebalance_freq}回測平均 Sharpe:", df_bt['sharpe'].mean().round(2))
    else:
        print("\n[錯誤] 無法計算 Sharpe：回測結果為空或缺少 'sharpe' 欄位。")
    print("line：", lineno())
    df_90 = model.backtest(window_length=90)
    print("line：", lineno())
    df_180 = model.backtest(window_length=180)
    print("line：", lineno())
    df_252 = model.backtest(window_length=252)
    print("line：", lineno())
    ## 如果有 market_cond 欄位，才做分組統計；否則提示警告
    #if 'market_cond' in df_252.columns:
    #    stats = df_252.groupby('market_cond')[['sharpe', 'drawdown']].mean()
    #    print("[資訊] 各市場狀態下的 Sharpe 與回撤平均值：")
    #    print(stats)
    #else:
    #    print("[警告] 無法進行市場狀態分組：DataFrame 中沒有 'market_cond' 欄位")
    #df_252.groupby('market_cond')[['sharpe', 'drawdown']].mean()
        
    script_path = sys.argv[0]
    script_name = os.path.basename(script_path)
    script_base = os.path.splitext(script_name)[0]
    # 結果檔直接放在 OUTPUT_ROOT 底下
    result_filename = f"{script_base}-result.txt"
    result_path = os.path.join(OUTPUT_ROOT, result_filename)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=== 最佳化結果與回測摘要 ===\n")
        # 範例：寫入 sharpe 平均、weights JSON、performance

        # 寫入結果檔時，同樣檢查回測資料
        if not df_bt.empty and 'sharpe' in df_bt.columns:
            f.write(f"回測平均 Sharpe: {df_bt['sharpe'].mean().round(2)}\n")
        else:
            f.write("回測平均 Sharpe: N/A\n")
        f.write("最終權重：\n")
        json.dump(model.weights, f, ensure_ascii=False, indent=2)
        f.write("\nPerformance:\n")
        # 如果 optimize 未成功產生 performance，則填 None，避免 TypeError
        if hasattr(model, 'performance') and model.performance is not None:
            exp_ret, vol, sharpe = model.performance
        else:
            exp_ret = vol = sharpe = None
        json.dump({
            "exp_ret": exp_ret,
            "vol": vol,
            "sharpe": sharpe
        }, f, ensure_ascii=False, indent=2)
    print(f"[資訊] 已寫入結果檔：{result_path}")
    src_prompt = Path('default_prompt.txt')
    dst_prompt = Path(OUTPUT_ROOT) / 'default_prompt.txt'
    if src_prompt.exists():
        shutil.copy(src_prompt, dst_prompt)
        print(f"已複製 default_prompt.txt 到 {dst_prompt}")
    else:
        print("警告：找不到 default_prompt.txt，請確認檔案位置")
    