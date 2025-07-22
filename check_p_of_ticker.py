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

def get_close_series(ticker, period="6mo", field: str = "close") -> pd.Series:
    """
    下載指定 ticker 的欄位序列。
    Parameters
    ----------
    ticker : str
    period : str  e.g. '6mo', '1y'
    field  : str  'close' (預設) | 'volume' | 任何 yfinance 回傳欄位名
    Returns
    -------
    pandas.Series  去除 NA
    """
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)

    field = field.lower()
    if field in {"close", "adj close", "adj_close", "price"}:
        if "Adj Close" in df.columns:
            s = df[["Adj Close"]]        # 先取成 DataFrame 以統一流程
        elif "Close" in df.columns:
            s = df[["Close"]]
        else:
            raise KeyError(f"{ticker} 找不到價格欄位")
    elif field == "volume":
        s = df[["Volume"]]
    elif field in df.columns:
        s = df[[field]]
    else:
        raise KeyError(f"{ticker} 無欄位 {field}")

    # 🆕─── **強制壓成 1‑D Series** ──────────────────────────
    s = s.squeeze("columns")            # DataFrame → Series；若已是 Series 不變
    if isinstance(s, np.ndarray):       # 萬一還是 ndarray
        s = pd.Series(s.ravel(), index=df.index[: len(s.ravel())])
    # ─────────────────────────────────────────────────────

    return s.dropna()


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
        #print(f"從 yfinance 讀入 {len(close)} 筆收盤價")

    # 印出最後 5 筆確認
    #print("最後五筆收盤價：")
    #print(close.tail())

    # 2. 計算過去 3 個月（約 63 交易日）的日報酬率
    window = args.window
    returns = close.pct_change().dropna()
    recent_returns = returns[-window:]
    mu_daily_raw = recent_returns.mean()
    sigma_daily_raw = recent_returns.std(ddof=1)
    # ── 將 pandas.Series ‚μ‘, ‚σ‘ 轉純量，避免後面 numpy 運算跑回 pandas ──
    mu_daily    = mu_daily = (float(mu_daily_raw.iloc[0]) if isinstance(mu_daily_raw, pd.Series)
             else float(mu_daily_raw))
    sigma_daily = (float(sigma_daily_raw.iloc[0]) if isinstance(sigma_daily_raw, pd.Series)
                else float(sigma_daily_raw))

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
    if Increase >= 1:   # 上漲事件
        prob = 1 - norm.cdf((np.log(Increase)-mu_T)/sigma_T)
    else:               # 下跌事件
        prob =      norm.cdf((np.log(Increase)-mu_T)/sigma_T)  # 左尾
    #prob = 1 - norm.cdf((threshold - mu_T) / sigma_T)
    # ─── 確保 prob 是純量 float ───
    prob      = float(prob)   # 或者用 prob = prob.item()
    #print(f"假設對數常態，三個月內漲幅≥10% 機率 ≈ {prob*100:.2f}%")
    
    # 進階：蒙地卡羅模擬 np.log
    simulations = 100_000
    Z = np.random.randn(simulations, T)
    S_T = np.exp(np.cumsum((mu_daily - 0.5*sigma_daily**2) + sigma_daily * Z, axis=1))
    # 模擬終值相對變動
    final_returns = S_T[:,-1]
    if Increase >= 1:   # 上漲事件
        mc_prob = np.mean(final_returns >= Increase)
    else:               # 下跌事件
        mc_prob = np.mean(final_returns <= Increase)
    #mc_prob = np.mean(final_returns >= Increase)
    #print(f"蒙地卡羅模擬三個月內漲幅≥10% 機率 ≈ {mc_prob*100:.2f}%")
    #print(
    #
    #f"  {(args.source)}: 讀入 {len(close)} 筆收盤價. 估"
    #f"{args.T:02d} 交易日內漲幅 ≥ "
    #f"{((args.Increase-1)*100):+06.2f}% 機率: 對數常態 ≈ "
    #f"{prob*100:05.2f}%，蒙地卡羅 ≈ {mc_prob*100:05.2f}%"
    #)
    # 1. 先計算百分比
    pct = abs(args.Increase - 1) * 100          # 5.0, 10.0, 30.0 …

    # 2. 決定描述文字與不等號方向
    if args.Increase >= 1:                      # ➕ 上漲事件
        event_desc = f"漲幅 ≥ +{pct:05.2f}%"
    else:                                       # ➖ 下跌事件
        event_desc = f"跌幅 ≤ -{pct:05.2f}%"

    # 3. 單行輸出
    print(
        f"  {(args.source)}: 讀入 {len(close)} 筆收盤價. "
        f"估{args.T:02d} 交易日內{event_desc} 機率: "
        f"對數常態 ≈ {prob*100:05.2f}%，蒙地卡羅 ≈ {mc_prob*100:05.2f}%"
    )
    #print(f"    {(args.source)}: 對數常態，{(args.T)}交易日數內漲幅 ≥  {(args.Increase-1)*100:.2f}% 機率 ≈ {prob*100:.2f}%, 蒙地卡羅模擬 {(args.T)}交易日數內漲幅≥{(args.Increase-1)*100:.2f}% 機率 ≈ {mc_prob*100:.2f}%")
    
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
    args.period='60mo'
    args.window=63
    performance=0.3
    args.Increase=1+performance
    args.T=63
    check_p_of_ticker(args)
    
    args.Increase=1+(performance/2)
    args.T=63
    check_p_of_ticker(args)
    
    args.Increase=1+(performance/2)
    args.T=42
    check_p_of_ticker(args)
    
    args.Increase=1+(performance/2)
    args.T=21
    check_p_of_ticker(args)
    
    args.Increase=1-performance
    args.T=63
    check_p_of_ticker(args)
    
    args.Increase=1-(0.1)
    args.T=21
    check_p_of_ticker(args)
    
    args.Increase=1-(0.1)
    args.T=10
    check_p_of_ticker(args)
    
    args.Increase=1-(0.05)
    args.T=5
    check_p_of_ticker(args)
        
    args.Increase=1-(0.05)
    args.T=3
    check_p_of_ticker(args)
            
    args.Increase=1-(0.05)
    args.T=1
    check_p_of_ticker(args)
    