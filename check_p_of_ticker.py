import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
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


def get_close_from_txt(filepath, sep='\t', date_col=0, price_col=1):
    df = pd.read_csv(filepath, sep=sep, header=None,
                     parse_dates=[date_col],
                     usecols=[date_col, price_col])
    df.columns = ['Date', 'Close']
    df.set_index('Date', inplace=True)
    series = df['Close'].dropna()
    #print(f"從 '{filepath}' 讀入 {len(series)} 筆收盤價")  # 確認筆數
    return series
    

def check_p_of_ticker(args):
    # 判斷來源
    # 1. 下載過去 6 個月的日線收盤價
    if args.source.lower().endswith('.txt'):
        close = get_close_from_txt(args.source, sep=args.sep)
    else:
        close = get_close_series(args.source, period=args.period)
        print(f"從 yfinance 讀入 {len(close)} 筆收盤價")

    # 印出最後 5 筆確認
    #print("最後五筆收盤價：")
    #print(close.tail())

    # 2. 計算過去 3 個月（約 63 交易日）的日報酬率
    window = args.window
    returns = close.pct_change().dropna()
    recent_returns = returns[-window:]
    mu_daily = recent_returns.mean()
    sigma_daily = recent_returns.std(ddof=1)
    # ── 將 pandas.Series ‚μ‘, ‚σ‘ 轉純量，避免後面 numpy 運算跑回 pandas ──
    mu_daily    = float(mu_daily.iloc[0])
    sigma_daily = float(sigma_daily.iloc[0])

    # 3. 描述近期趨勢
    total_return_3m = (close.iloc[-1] / close.iloc[-window] - 1) * 100
    
    # 4. 估算三個月內上漲 10% 的機率
    T = args.T  # 交易日數
    # 計算門檻 代表漲幅 10%
    Increase = args.Increase
    
    
    # 假設對數報酬常態分布，計算 P(S_T/S_0 ≥ 1.1)
    # ln(S_T/S_0) ~ N((μ_daily - 0.5 σ_daily^2)*T, σ_daily^2 * T)
    mu_T = (mu_daily - 0.5 * sigma_daily**2) * T
    sigma_T = sigma_daily * np.sqrt(T)
    threshold = np.log(Increase)
    prob = 1 - norm.cdf((threshold - mu_T) / sigma_T)
    # ─── 確保 prob 是純量 float ───
    prob      = float(prob)   # 或者用 prob = prob.item()
    #print(f"假設對數常態，三個月內漲幅≥10% 機率 ≈ {prob*100:.2f}%")
    
    # 進階：蒙地卡羅模擬
    simulations = 100_000
    Z = np.random.randn(simulations, T)
    S_T = np.exp(np.cumsum((mu_daily - 0.5*sigma_daily**2) + sigma_daily * Z, axis=1))
    # 模擬終值相對變動
    final_returns = S_T[:,-1]
    mc_prob = np.mean(final_returns >= Increase)
    #print(f"蒙地卡羅模擬三個月內漲幅≥10% 機率 ≈ {mc_prob*100:.2f}%")
    print(f"    {(args.source)}: 對數常態，{(args.T)}交易日數內漲幅≥{(args.Increase-1)*100:.2f}% 機率 ≈ {prob*100:.2f}%, 蒙地卡羅模擬 {(args.T)}交易日數內漲幅≥{(args.Increase-1)*100:.2f}% 機率 ≈ {mc_prob*100:.2f}%")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Stock probability estimation\n'
                    '用法範例: python stock.py --source 2317.TW\n'
                    '      python stock.py --source prices.txt --sep ,'
    )
    parser.add_argument('--source', required=True,
                        help='股票代號 (e.g. 2317.TW) 或 .txt 檔路徑')
    parser.add_argument('--period', default='6mo',
                        help='yfinance 下載區間，例如 6mo')
    parser.add_argument('--sep', default='\t',
                        help='.txt 檔案的分隔符號，預設為 TAB')
    parser.add_argument('--window', type=int, default=63,
                        help='計算報酬的交易日視窗長度')
    parser.add_argument('--T', type=float, default=63,
                        help='預期報酬的交易日視窗長度')
    args = parser.parse_args()

    args.sep='\t'
    args.period='6mo'
    args.window=63
    performance=0.3
    args.Increase=1+performance
    args.T=63
    check_p_of_ticker(args)
    
    args.Increase=1+(performance/2)
    args.T=21
    check_p_of_ticker(args)
    
    args.Increase=1-performance
    args.T=63
    check_p_of_ticker(args)
    
    args.Increase=1-(performance/2)
    args.T=21
    check_p_of_ticker(args)
    
    args.Increase=1-(performance/2)
    args.T=10
    check_p_of_ticker(args)
    
    args.Increase=1-(performance/2)
    args.T=5
    check_p_of_ticker(args)
        
    args.Increase=1-(performance/2)
    args.T=3
    check_p_of_ticker(args)
            
    args.Increase=1-(performance/2)
    args.T=1
    check_p_of_ticker(args)
    