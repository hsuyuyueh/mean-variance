# -*- coding: utf-8 -*-
"""
check_p_of_ticker_V6.py  – 2025‑07‑22
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

一次計算 **6 種機率模型**（Normal、Student‑t、Bootstrap、GARCH‑t、Jump‑Diffusion、Monte Carlo）並排版輸出。

CLI 參數說明
------------
* `--source`     股票代號，例如 `2317.TW`
* `--Increase`   門檻倍率 `>1` 代表漲幅，`<1` 代表跌幅
* `--T`          交易日數（預設 63）
* `--period`     yfinance 抓取區間（預設 60mo）
* `--method`     `all`（預設值）或單獨指定
                 `normal | student_t | bootstrap | garch_t | jump_diff | MonteCarlo`

範例
----
```bash
python3 check_p_of_ticker_V6.py \
    --source 2317.TW --Increase 0.95 --T 5 --period 60mo      # 六法同時

python3 check_p_of_ticker_V6.py \
    --source 2317.TW --method garch_t --Increase 0.95 --T 5   # 只跑 GARCH‑t
```
"""
from __future__ import annotations

import argparse
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, t

try:
    from arch import arch_model  # type: ignore
except ImportError:  # pragma: no cover
    arch_model = None  # 在缺少 arch 時退化提醒

# ──────────────────── 資料處理 ─────────────────────────────

def fetch_close_series(ticker: str, period: str = "12mo") -> pd.Series:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].dropna()


def log_returns(price: pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]
    return np.log(price / price.shift(1)).dropna().to_numpy().flatten()

# ──────────────────── 機率計算函式 ─────────────────────────

def prob_lognormal(mu_d: float, sigma_d: float, T: int, k: float) -> float:
    mu_T = (mu_d - 0.5 * sigma_d**2) * T
    sigma_T = sigma_d * np.sqrt(T)
    z = (np.log(k) - mu_T) / sigma_T
    return 1 - norm.cdf(z) if k >= 1 else norm.cdf(z)


def prob_student_t(log_ret: np.ndarray, T: int, k: float) -> float:
    df_, loc, scale = t.fit(log_ret)
    loc_T = loc * T
    scale_T = scale * np.sqrt(T)
    z = (np.log(k) - loc_T) / scale_T
    return 1 - t.cdf(z, df_) if k >= 1 else t.cdf(z, df_)


def mc_geometric_bm(mu_d: float, sigma_d: float, T: int, k: float, sims: int = 100_000) -> float:
    Z = np.random.randn(sims, T)
    log_path = np.cumsum((mu_d - 0.5 * sigma_d**2) + sigma_d * Z, axis=1)
    ratio = np.exp(log_path[:, -1])
    return np.mean(ratio >= k) if k >= 1 else np.mean(ratio <= k)


def bootstrap_prob(log_ret: np.ndarray, T: int, k: float, sims: int = 100_000) -> float:
    paths = np.random.choice(log_ret, size=(sims, T), replace=True).sum(axis=1)
    ratio = np.exp(paths)
    return np.mean(ratio >= k) if k >= 1 else np.mean(ratio <= k)

# ────────── GARCH‑t ──────────

def prob_garch_t(log_ret: np.ndarray, T: int, k: float, sims: int = 10_000) -> float:
    if arch_model is None:
        raise RuntimeError("arch 未安裝，無法執行 garch_t")
    scaled = log_ret * 100
    am = arch_model(scaled, vol="Garch", p=1, q=1, dist="t", mean="Zero")
    res = am.fit(disp="off")
    sim = res.forecast(horizon=T, method="simulation", simulations=sims, random_state=42)
    cum_paths = sim.simulations.values.sum(axis=0) / 100  # 還原百分比
    ratio = np.exp(cum_paths)
    return np.mean(ratio >= k) if k >= 1 else np.mean(ratio <= k)

# ────────── Jump‑Diffusion ──────────

def _estimate_jump_params(log_ret: np.ndarray) -> Tuple[float, float, float]:
    sigma_d = log_ret.std(ddof=1)
    jumps = log_ret[np.abs(log_ret) > 3 * sigma_d]
    lam = len(jumps) / len(log_ret)
    if lam == 0:
        return 0.0, 0.0, 0.0
    return lam, jumps.mean(), (jumps.std(ddof=1) if len(jumps) > 1 else sigma_d)


def prob_jump_diffusion(log_ret: np.ndarray, T: int, k: float, sims: int = 100_000) -> float:
    mu_d, sigma_d = log_ret.mean(), log_ret.std(ddof=1)
    lam, mu_j, sigma_j = _estimate_jump_params(log_ret)
    if lam == 0:
        return mc_geometric_bm(mu_d, sigma_d, T, k, sims)

    drift = mu_d - 0.5 * sigma_d**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    Z = np.random.randn(sims, T)
    N = np.random.poisson(lam, size=(sims, T))
    J = np.random.normal(mu_j, sigma_j, size=(sims, T)) * N
    log_path = np.cumsum(drift + sigma_d * Z + J, axis=1)
    ratio = np.exp(log_path[:, -1])
    return np.mean(ratio >= k) if k >= 1 else np.mean(ratio <= k)

# ──────────────────── 輔助 ───────────────────────────────

def event_desc(k: float) -> str:
    pct = abs(k - 1) * 100
    return f"漲幅 ≥ +{pct:05.2f}%" if k >= 1 else f"跌幅 ≤ -{pct:05.2f}%"

# ──────────────────── 公用 API  ───────────────────────────
def compute_probabilities(
    ticker: str,
    increase: float,
    T: int,
    period: str = "12mo",
    methods: list[str] | None = None,
) -> dict[str, float]:
    """
    回傳 {method: prob}，不印任何字，給外部程式調用。
    """
    close    = fetch_close_series(ticker, period)
    log_ret  = log_returns(close)
    mu_d, σ_d = log_ret.mean(), log_ret.std(ddof=1)

    mapping: dict[str, Callable[[], float]] = {
        "normal":     lambda: prob_lognormal(mu_d, σ_d, T, increase),
        "student_t":  lambda: prob_student_t(log_ret, T, increase),
        "bootstrap":  lambda: bootstrap_prob(log_ret, T, increase),
        "garch_t":    lambda: prob_garch_t(log_ret, T, increase),
        "jump_diff":  lambda: prob_jump_diffusion(log_ret, T, increase),
        "MonteCarlo": lambda: mc_geometric_bm(mu_d, σ_d, T, increase),
    }

    if methods is None or methods == ["all"]:
        methods = list(mapping.keys())

    return {m: mapping[m]() for m in methods}
    
# ──────────────────── 主流程 ─────────────────────────────

def run(args: argparse.Namespace) -> None:
    close = fetch_close_series(args.source, period=args.period)
    log_ret = log_returns(close)
    mu_d, sigma_d = log_ret.mean(), log_ret.std(ddof=1)

    mapping: Dict[str, Tuple[Callable[..., float], str]] = {
        "normal":     (lambda: prob_lognormal(mu_d, sigma_d, args.T, args.Increase), "normal"),
        "student_t":  (lambda: prob_student_t(log_ret, args.T, args.Increase), "student_t"),
        "bootstrap":  (lambda: bootstrap_prob(log_ret, args.T, args.Increase), "bootstrap"),
        "garch_t":    (lambda: prob_garch_t(log_ret, args.T, args.Increase), "garch_t"),
        "jump_diff":  (lambda: prob_jump_diffusion(log_ret, args.T, args.Increase), "jump_diff"),
        "MonteCarlo": (lambda: mc_geometric_bm(mu_d, sigma_d, args.T, args.Increase), "MonteCarlo"),
    }

    # 要執行的模型列表
    methods = list(mapping.keys()) if args.method == "all" else [args.method]

    results: Dict[str, float] = {}
    for m in methods:
        results[m] = mapping[m][0]()

    # ────────── 輸出 ──────────
    print(f"{args.source}: 讀 {len(close)} 筆價格. 估{args.T:02d}交易日內{event_desc(args.Increase)} 機率：")
    header = "  ".join(name.ljust(11) for name in mapping.keys())
    print(header)
    row = "       ".join(f"{results.get(name, float('nan'))*100:05.2f}%" if name in results else "     -   "
                      for name in mapping.keys())
    print(row)

# ──────────────────── CLI ───────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser("6‑Method Probability Estimator")
    ap.add_argument("--source", required=True, help="股票代號，如 2317.TW")
    ap.add_argument("--Increase", type=float, default=1.1, help="門檻倍率，>1 為漲幅，<1 為跌幅")
    ap.add_argument("--T", type=int, default=63, help="交易日數")
    ap.add_argument("--period", default="12mo", help="yfinance 抓取區間")
    ap.add_argument("--method", choices=[
        "all", "normal", "student_t", "bootstrap", "garch_t", "jump_diff", "MonteCarlo"],
        default="all", help="演算法 (all = 六法並列)")
    args = ap.parse_args()
    
    args.T=5
    args.Increase=0.95
    run(args)
    args.T=5
    args.Increase=0.90
    run(args)
    args.T=10
    args.Increase=0.90
    run(args)
    args.T=60
    args.Increase=0.90
    run(args)
    print("========")
    args.T=5
    args.Increase=1.05
    run(args)
    args.T=10
    args.Increase=1.05
    run(args)
    args.T=20
    args.Increase=1.05
    run(args)
    args.T=60
    args.Increase=1.2
    run(args)
