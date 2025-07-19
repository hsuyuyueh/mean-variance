#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nest_asyncio
import pandas as pd
import yfinance as yf
import json
import os
import matplotlib.pyplot as plt
import argparse, sys, json
from pathlib import Path
import shutil

from datetime import datetime, timedelta
from pypfopt.risk_models import exp_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.objective_functions import L2_reg
from scipy.stats import norm
from tqdm import tqdm


from fetch_0056_components import fetch_0056_components
from fetch_0050_components import fetch_0050_components
from fetch_00713_components import fetch_00713_components
from fetch_US_berkshire_components import fetch_US_berkshire_components
from fetch_US_harvard_components import fetch_US_harvard_components
from fetch_US_SPY_components import fetch_US_SPY_components
from my_ai_module import gpt_contextual_rating



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
    except:
        return None

def fetch_price_alphaav(symbol, start, end, api_key, pause=12):
    # 先查 SQLite 快取
    cur = conn.execute(
        "SELECT date, adj_close FROM price WHERE symbol=? AND date BETWEEN ? AND ?",
        (symbol, start, end)
    )
    rows = cur.fetchall()
    if rows:
        import pandas as pd
        s = pd.Series({r[0]: r[1] for r in rows})
        return s.sort_index()

    # 否則呼叫 Alpha Vantage
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full"
    )
    r = requests.get(url)
    data = r.json().get("Time Series (Daily)", {})
    import pandas as pd
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    s = df["5. adjusted close"].loc[start:end].astype(float)
    # 寫入快取
    for dt, val in s.items():
        conn.execute(
            "INSERT OR IGNORE INTO price(symbol,date,adj_close) VALUES(?,?,?)",
            (symbol, dt.strftime("%Y-%m-%d"), float(val))
        )
    conn.commit()
    time.sleep(pause)
    return s

def fetch_price(symbols, start, end, av_api_key):
    prices = {}
    for sym in symbols:
        p = fetch_price_yahoo(sym, start, end)
        if p is None or p.empty:
            p = fetch_price_alphaav(sym, start, end, av_api_key)
        prices[sym] = p
    # 回傳一個 pandas.DataFrame
    import pandas as pd
    return pd.DataFrame(prices)


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

    def fetch_fundamentals(self):
        cache_file = os.path.join(OUTPUT_ROOT, f"fundamentals_{self.RUN_DAT}.json")
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
        cache_file = os.path.join(self.OUTPUT_ROOT, f"default_mu_cache_{self.RUN_DAT}.json")
        # —— Step 1: 本地計算 historical μ —— 
        import numpy as np
        log_ret = np.log(self.prices / self.prices.shift(1)).dropna()
        mu_local = log_ret.mean() * 252
        # 儲存本地 μ 到檔案，方便追蹤
        local_mu_path = os.path.join(self.OUTPUT_ROOT, f"local_mu_{self.RUN_DAT}.json")
        mu_local.to_json(local_mu_path, force_ascii=False, indent=2)
        print(f"[INFO] 已儲存本地 historical μ 至 {local_mu_path}")

        # —— Step 2: 本地計算技術指標 —— 
        # 若價格資料為空，跳過計算並預設 None
        if self.prices.empty:
            tech_indicators = {
                "ma5":       None,
                "macd":      None,
                "kd_k":      None,
                "kd_d":      None,
                "year_line": None,
            }
        else:
            # 使用完整 DataFrame 計算各指標
            adj = self.prices
            ma5       = adj.rolling(window=5).mean().iloc[-1]
            ema12     = adj.ewm(span=12).mean()
            ema26     = adj.ewm(span=26).mean()
            macd      = (ema12 - ema26).iloc[-1]
            low9      = adj.rolling(9).min()
            high9     = adj.rolling(9).max()
            kd_k      = ((adj - low9) / (high9 - low9) * 100).iloc[-1]
            kd_d      = kd_k.rolling(3).mean().iloc[-1]
            year_line = adj.rolling(window=252).mean().iloc[-1]
            tech_indicators = {
                "ma5":       round(ma5,       2) if pd.notna(ma5) else None,
                "macd":      round(macd,      4) if pd.notna(macd) else None,
                "kd_k":      round(kd_k,      2) if pd.notna(kd_k) else None,
                "kd_d":      round(kd_d,      2) if pd.notna(kd_d) else None,
                "year_line": round(year_line, 2) if pd.notna(year_line) else None,
            }

        # —— Step 3: 呼叫 AI，將技術指標也傳給 gpt_contextual_rating —— 
        self.mu_final = gpt_contextual_rating(
            tickers=self.tickers,
            base_mu=mu_local.to_dict(),
            tech_indicators=tech_indicators,
            force=False,
            OUTPUT_ROOT=self.OUTPUT_ROOT
        )

    def optimize(self):
        sigma = exp_cov(self.prices, span=180)
        # 加入微小對角 jitter，確保 covariance matrix 為正定
        import numpy as np
        sigma += np.eye(sigma.shape[0]) * 1e-4
        common = self.mu_final.index.intersection(sigma.index)
        # 若沒有任何資產交集，無法進行最佳化，提前返回
        if len(common) == 0:
            print("[錯誤] 無法進行最佳化：mu 與 cov 的交集為空，請確認是否成功取得股價與預測。")
            return
        mu = self.mu_final.loc[common]
        sigma = sigma.loc[common, common]
        
        ef1 = EfficientFrontier(mu, sigma, weight_bounds=(0, 1))
        # 嘗試以使用者設定的無風險利率計算最大化 Sharpe 組合
        try:
            ef1.max_sharpe(risk_free_rate=self.rf_rate)
        except ValueError as e:
            # 當所有資產預期報酬都 ≤ 無風險利率時，改用無風險利率=0 重算
            print(f"[警告] {e}，改用無風險利率=0 重新計算 max_sharpe")
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
                except:
                    pass
                print(f"{tk}: {w:.2%}, μ={mu.get(tk,0):.2%}, Sector={sector}")
        print(f"\n預期年化報酬: {self.performance[0]:.2%}, 年化波動率: {self.performance[1]:.2%}, Sharpe: {self.performance[2]:.2f}, P: {prob:.2f}\n")

    def estimate_portfolio_beta(self):
        idx_map = {'TW':'^TWII', 'US':'^GSPC', 'JP':'^N225'}
        benchmark = yf.download(idx_map[self.market], period="1y", auto_adjust=False)['Adj Close']\
                  .pct_change().dropna()
        betas = {}
        for tk in self.tickers:
            try:
                stk  = yf.download(tk, period="1y", auto_adjust=False)['Adj Close']\
                          .pct_change().dropna()
                df2  = pd.concat([stk, benchmark], axis=1).dropna()
                betas[tk] = df2.cov().iloc[0,1] / df2.iloc[:,1].var()
            except:
                betas[tk] = 1.0
        return round(sum(self.weights.get(tk,0)*betas.get(tk,1) for tk in self.tickers), 2)

    def save_weights(self, filepath=None):
        path = filepath or os.path.join(OUTPUT_ROOT, "current_portfolio.json")
        with open(path, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def backtest(self, rebalance_freq='2W'):
        results = []
        # 若價格資料為空，無法進行回測，直接返回空 DataFrame
        if self.prices.empty:
            print("[錯誤] 無法回測：價格資料為空，請確認是否成功取得股價資料。")
            return pd.DataFrame()
        # 生成回測日期序列
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
    if args.market == "US":                       # 只清美股
        tickers = sorted({ _normalize_us_ticker(t) for t in tickers })
    start   = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end     = datetime.today().strftime('%Y-%m-%d')
    
    # 改用備援方案取得價格
    prices = fetch_price(
        tickers,
        start,
        end,
        av_api_key="你的 Alpha Vantage API KEY"
    ).dropna(axis=1, how="all")

    missing = set(tickers) - set(prices.columns)
    if missing:
        print(f"[警告] 無法取得以下股價：{sorted(missing)}")

    
    rf_table = {'TW':0.015, 'US':0.045, 'JP':0.002}
    

    
    model   = AiMeanVariancePortfolio(
                 tickers, prices,
                 market=args.market,              # 傳進去
                 OUTPUT_ROOT=OUTPUT_ROOT,
                 RUN_DATE=RUN_DATE,
                 profile_level='P4'
              )
    model.rf_rate = rf_table[args.market]         # 依市場覆寫無風險利率
    model.fetch_fundamentals()
    model.build_mu()
    model.optimize()
    model.save_weights()
    df_bt = model.backtest()
    # 保護性檢查：確保有回測結果且包含 sharpe 欄位才列印，否則提示錯誤
    if not df_bt.empty and 'sharpe' in df_bt.columns:
        print("\n回測平均 Sharpe:", df_bt['sharpe'].mean().round(2))
    else:
        print("\n[錯誤] 無法計算 Sharpe：回測結果為空或缺少 'sharpe' 欄位。")

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
    