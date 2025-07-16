# mean-variance

---

## 🧠 模組解說

### 1. `gpt_contextual_rating()`
- 利用 GPT 分析新聞/基本面，預測個股 3 個月報酬率 μ。

### 2. `apply_momentum_adjustment()`
- 加權放大近期 60 日內強勢個股，強化動能因子。
- 越漲得多，μ 放大越多。

### 3. `apply_mu_confidence_adjustment()`
- 使用波動率懲罰 μ：波動高的股票，其 AI μ 會被打折，避免誤配高風險資產。

### 4. `optimize_portfolio()`
- 使用 PyPortfolioOpt 建構 Max Sharpe 投資組合。
- 加入 `L2 正則化` 讓權重不極端。

### 5. `compute_market_risk_factor()`
- 動態評估市場風險因子：
    - VIX 上升 → 降低曝險
    - JNK 下跌 → 降低曝險
    - 回撤超過 15% → 降低曝險
- 綜合成一個係數 (0.4 ~ 1.0)

### 6. `apply_risk_controls()`
- 單檔持股上限 20%
- 若權重偏差不大則不調整
- 若市場風險高 → 整體降槓桿

---

## 📈 輸出報表

### 結果包含：
- Sharpe, 預期報酬, 波動率
- 各檔資產配重
- Top 5 配重股票之：
    - μ 預測值
    - 所屬產業 (sector)
    - 配置金額

---


---

## 🧯 客戶風險等級與 β 控管

此模型支援根據不同投資者風險承受度進行 **β 風險值限制**，範例如下：

| 等級 | 對應客戶        | 可接受 β（市場風險）上限 |
|------|------------------|--------------------------|
| P1   | C1 保守型         | β ≤ 0.8                  |
| P2   | C2 穩健型         | β ≤ 1.0                  |
| P3   | C3 成長型（預設） | β ≤ 1.3                  |
| P4   | C4 積極型         | 無上限                   |

透過參數設定：

```python
model = AiMeanVariancePortfolio(tickers, prices, profile_level='P2')
```

系統將在報告中印出投組 β 並自動提示風險是否超標，協助投資顧問做出適當建議。

---

## ✅ 技術細節
- 無風險利率動態抓取自 ^IRX (7日平均)
- ETF 成分股來自 0050 + 0056，自動排除「289」開頭金融股
- 使用 `yfinance` 補充產業分類與報價資料

---

## 🔚 結語
這個版本為策略性 AI + Mean-Variance 模型的進階組合，適合進行長期資金配置與波動控制。你可以透過調整：

- μ horizon
- momentum weight
- 信心懲罰係數
- 曝險門檻 (VIX/JNK/Drawdown)

來設計出不同風格的 AI 投資組合。

"""

🧠 模組解說
1. gpt_contextual_rating()
利用 GPT 分析新聞/基本面，預測個股 3 個月報酬率 μ。

2. apply_momentum_adjustment()
加權放大近期 60 日內強勢個股，強化動能因子。

越漲得多，μ 放大越多。

3. apply_mu_confidence_adjustment()
使用波動率懲罰 μ：波動高的股票，其 AI μ 會被打折，避免誤配高風險資產。

4. optimize_portfolio()
使用 PyPortfolioOpt 建構 Max Sharpe 投資組合。

加入 L2 正則化 讓權重不極端。

5. compute_market_risk_factor()
動態評估市場風險因子：

VIX 上升 → 降低曝險

JNK 下跌 → 降低曝險

回撤超過 15% → 降低曝險

綜合成一個係數 (0.4 ~ 1.0)

6. apply_risk_controls()
單檔持股上限 20%

若權重偏差不大則不調整

若市場風險高 → 整體降槓桿

📈 輸出報表
結果包含：
Sharpe, 預期報酬, 波動率

各檔資產配重

Top 5 配重股票之：

μ 預測值

所屬產業 (sector)

配置金額

✅ 技術細節
無風險利率動態抓取自 ^IRX (7日平均)

ETF 成分股來自 0050 + 0056，自動排除「289」開頭金融股

使用 yfinance 補充產業分類與報價資料