import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

# --- 新增：設定 matplotlib 支援中文 ---
# 在 Windows 上通常有 "Microsoft JhengHei"；Linux/macOS 需先安裝對應字體並改成字體名稱
mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['axes.unicode_minus'] = False  # 處理負號顯示

def get_close_series(ticker, period="6mo"):
    # 方式一：若需要 Adj Close，設定 auto_adjust=False
    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    # 若仍想用 auto_adjust=True 版本，註解上行並改用下面這行：
    # data = yf.download(ticker, period=period, progress=False)
    
    # 彈性取用價格欄位
    if 'Adj Close' in data.columns:
        close = data['Adj Close']
    elif 'Close' in data.columns:
        close = data['Close']
    else:
        raise KeyError(f"{ticker} 資料中找不到 'Adj Close' 或 'Close' 欄位")
    
    return close.dropna()

if __name__ == "__main__":
    # 1. 下載過去 6 個月的日線收盤價
    ticker = "2885.TW"
    close = get_close_series(ticker)
    print(f"{ticker} 最後五筆收盤價：\n", close.tail())
    #data = yf.download(ticker, period="6mo", progress=False, auto_adjust=False)
    #close = data['# 如果 close 還是 DataFrame，就轉成 Series
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    
    print(f"{ticker} 最後五筆收盤價：\n", close.tail())
    
    # 2. 計算過去 3 個月（約 63 交易日）的日報酬率
    window = 63
    returns = close.pct_change().dropna()
    recent_returns = returns[-window:]
    mu_daily = recent_returns.mean()
    sigma_daily = recent_returns.std(ddof=1)
    
    # 3. 描述近期趨勢
    total_return_3m = (close.iloc[-1] / close.iloc[-window] - 1) * 100
    ma20 = close.rolling(window=20).mean()
    ma60 = close.rolling(window=60).mean()
    
    print(f"過去 3 個月總漲跌幅：{total_return_3m:.2f}%")
    print(f"日均報酬率 μ_daily：{mu_daily:.4f}, 日波動率 σ_daily：{sigma_daily:.4f}")
    
    # 畫圖觀察收盤價與移動平均
    ma20 = close.rolling(window=20, min_periods=1).mean()
    ma60 = close.rolling(window=60, min_periods=1).mean()
    plt.figure(figsize=(10,5))
    plt.plot(close.index, close, label='Close')
    plt.plot(ma20.index, ma20, label='MA20')
    plt.plot(ma60.index, ma60, label='MA60')
    plt.title("2885.TW 收盤價與移動平均")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 4. 估算三個月內上漲 10% 的機率
    T = 5  # 交易日數
    # 假設對數報酬常態分布，計算 P(S_T/S_0 ≥ 1.1)
    # ln(S_T/S_0) ~ N((μ_daily - 0.5 σ_daily^2)*T, σ_daily^2 * T)
    mu_T = (mu_daily - 0.5 * sigma_daily**2) * T
    sigma_T = sigma_daily * np.sqrt(T)
    threshold = np.log(1.1)
    prob = 1 - norm.cdf((threshold - mu_T) / sigma_T)
    
    print(f"假設對數常態，三個月內漲幅≥10% 機率 ≈ {prob*100:.2f}%")
    
    # 進階：蒙地卡羅模擬
    simulations = 100_000
    Z = np.random.randn(simulations, T)
    S_T = np.exp(np.cumsum((mu_daily - 0.5*sigma_daily**2) + sigma_daily * Z, axis=1))
    # 模擬終值相對變動
    final_returns = S_T[:,-1]
    mc_prob = np.mean(final_returns >= 1.1)
    print(f"蒙地卡羅模擬三個月內漲幅≥10% 機率 ≈ {mc_prob*100:.2f}%")
    