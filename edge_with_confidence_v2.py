# edge_with_confidence_v2.py
"""
Edge = α · P_lognormal  +  (1-α) · WinRate_condition

‒ P_lognormal  來自 check_p_of_ticker.calc_prob(...)
‒ WinRate_condition  來自歷史條件勝率表
"""

import os, json, datetime as dt
import numpy as np, pandas as pd

# ➜ 你自己的 util：抓收盤價、KD、成交量
from check_p_of_ticker import get_close_series
from check_p_of_ticker_V6 import compute_probabilities

try:
    import numpy as np
    # monkey-patch：補上 numpy.NaN 屬性以相容 pandas_ta
    setattr(np, "NaN", np.nan)
    import pandas_ta as ta      # 用來算 KD
except ImportError:
    raise ImportError("請：pip install pandas_ta")

# ---------- 全域參數（之後可搬到 YAML） ----------
ALPHA            = 0.6        # P 與 WinRate 的混合法
LOOKBACK_DAYS    = 252        # 勝率表回望長度
FWD_DAYS         = 5          # 勝率表計算『n 日後』報酬
VOL_MULTIPLIER   = 1.5        # 量能 > MA*n 判定爆量
OUTPUT_ROOT      = os.getenv("OUTPUT_ROOT", "./outputs")
# ------------------------------------------------

def _today_dir():
    today = dt.date.today().strftime("%Y%m%d")
    path  = f"{OUTPUT_ROOT}/{today}/edge_cache"
    os.makedirs(path, exist_ok=True)
    return path

def _cond_key(kd_cross: bool, vol_pump: bool) -> str:
    return f"KDcross={int(kd_cross)}|VOLpump={int(vol_pump)}"

# ===== 1. 產生 / 讀取 勝率表 =====
def build_winrate_table(ticker: str) -> dict:
    """回傳 {cond_key: {'count':int, 'win':int}}"""
    # (1) 下載 LOOKBACK_DAYS 的收盤價
    close = get_close_series(ticker, field="close")[-LOOKBACK_DAYS:]
    if close.empty:
        raise ValueError(f"{ticker} 沒抓到足夠價格資料")

    # ---------- 新增防呆 ---------- #
    if np.isscalar(close):                 # 只抓到一個數字？
        close = [close]                   # 先包進 list
    # -------------------------------- #

    df = pd.Series(close, name="close").to_frame()
    kd_df = ta.stoch(
                high = df["close"],   # 用同一序列當 high
                low  = df["close"],   # 用同一序列當 low
                close= df["close"],
                length=9
            )
    df["kd_k"] = kd_df.iloc[:, 0]     # 取 %K
    df["kd_d"] = kd_df.iloc[:, 1]     # 取 %D
    df["vol"]  = get_close_series(ticker, field="volume")   # 你的 util 要能抓 volume
    df["vol_ma"] = df["vol"].rolling(20).mean()

    # (2) 定義條件
    kd_cross   = (df["kd_k"].shift(1) < 20) & (df["kd_k"] > 20)
    vol_pump   = df["vol"] > df["vol_ma"] * VOL_MULTIPLIER
    cond       = kd_cross & vol_pump

    # (3) 未來 FWD_DAYS 報酬
    fwd_ret    = df["close"].pct_change(FWD_DAYS).shift(-FWD_DAYS)
    win        = fwd_ret > 0

    table = {
        _cond_key(True, True): {
            "count": int(cond.sum()),
            "win":   int((win[cond]).sum())
        }
    }
    # (4) 快取
    cache_fp = f"{_today_dir()}/{ticker}.json"
    json.dump(table, open(cache_fp, "w"), indent=2)
    return table

def _get_table(ticker: str) -> dict:
    cache_fp = f"{_today_dir()}/{ticker}.json"
    if os.path.exists(cache_fp):
        return json.load(open(cache_fp))
    return build_winrate_table(ticker)

# ===== 2. Edge 計算 =====
def calc_edge(ticker: str,
              kd_cross: bool,
              vol_pump: bool,
              p_log: float,
              alpha: float = ALPHA) -> float:
    tab = _get_table(ticker)
    key = _cond_key(kd_cross, vol_pump)
    hit = tab.get(key, {"count":0, "win":0})
    winrate = hit["win"]/hit["count"] if hit["count"] else 0.5
    return alpha * p_log + (1-alpha) * winrate

# ===== 3. 一行封裝供外部呼叫 =====
def get_edge_score(ticker: str, kd_series, vol_series) -> float:
    kd_cross  = (kd_series[-2] < 20) and (kd_series[-1] > 20)
    vol_pump  = vol_series[-1] > vol_series[-20:].mean() * VOL_MULTIPLIER
    # 使用對數常態模型計算未來 FWD_DAYS 日內「漲幅 ≥ 0%」的機率
    prob_dict = compute_probabilities(ticker, increase=1.0, T=FWD_DAYS)
    p_log = prob_dict.get("normal", 0.0)
    return calc_edge(ticker, kd_cross, vol_pump, p_log)
