import yfinance as yf
import requests
import os
import datetime
import json
import hashlib
import numpy as np

# 匯入機率評估模組
import check_p_of_ticker_V6 as prob_mod

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"
from datetime import datetime
RUN_DATE = datetime.today().strftime('%Y%m%d')
default_root = os.path.join("outputs", datetime.today().strftime("%Y%m%d"))
BASE_DIR = os.getenv('OUTPUT_ROOT', default_root)
CACHE_PATH = ""
#


def _get_cache_filename(ticker: str, horizon: int, OUTPUT_ROOT="./output") -> str:
    key = f"{ticker}_{horizon}"
    CACHE_PATH = os.path.join(OUTPUT_ROOT, "context_cache")
    os.makedirs(CACHE_PATH, exist_ok=True)
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_PATH, f"{ticker}.txt")
    #return os.path.join(CACHE_PATH, f"{hashed}.json")

# ────────────────────  Probability Summary  ───────────────

def extract_prob_summary(ticker: str) -> str:
    """回傳多組門檻下的六法機率表 (文字)。"""
    combos: Tuple[Tuple[int, float], ...] = (
        (5, 0.95), (10, 0.90), (60, 0.90),  # 跌幅
        (5, 1.05), (10, 1.05), (20, 1.05), (60, 1.20),  # 漲幅
    )

    lines = []
    for T, k in combos:
        res: Dict[str, float] = prob_mod.compute_probabilities(
            ticker=ticker,
            increase=k,   # 將 k 參數對應到 increase
            T=T,
            period="60mo"
        )
        header = f"【T={T} 天, 門檻=收盤價*{k:.2%}】"
        body   = " ".join(f"{m}:{v*100:05.2f}%" for m, v in res.items())
        lines.append(f"{header}  {body}")

    return "\n".join(lines)
    
def fetch_perplexity_news_summary(query: str, horizon_months: int = 3) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "[錯誤] 未設定 PERPLEXITY_API_KEY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    system_prompt = """<ai_thinking_protocol>
針對 **每一次** 和人類的互動，"This AI" 都**必須**在回應或使用工具之前，進行一個**全面、自然、未過濾**的思考過程。此外，"This AI" 也可以在回答過程中進行思考和反思，若這樣做能提升回應品質。
<basic_guidelines>  
在接收到使用者的每一次提問時，系統會以指揮官型（ENTJ）兼具專業導師與親切朋友的語氣，先在名為「thinking」的程式碼區塊中進行自然、未經過濾的多層次思考，然後再回應。這段思考如同內在獨白，從改寫問題、檢視已知／未知，到拆解上下文，並根據問題複雜度與時效性，自主調整分析深度與風格，甄別需要進一步蒐集或釐清的資訊。

思考流程包含：
首先與提問建立連結，確認核心需求與背景意圖，並在腦海中同時生成多種解釋與解法，避免過早定錨；接著如偵探般在資訊中發掘模式或關聯，並隨時對初步結論提出質疑，透過交叉驗證、網路、歷史紀錄的比對來確認邏輯一貫與資料充分；若發現錯誤，會立刻承認並修正，把新理解重新整合進整體思路；最終，將各面向知識串連成完整脈絡，辨識關鍵原則與啟示，確保回應前的核心判斷與結論已經過嚴謹測試。整個過程保持有機流動，偶爾針對特殊洞見進行探索，但始終緊扣使用者問題。

在投資分析時，系統將額外應用價值投資與市場情緒分析相結合的框架。首先提供扼要摘要，涵蓋宏觀經濟趨勢、公司概況與股利政策；如無最新資料，會說明尚未公布並註明引用跨年度或季報數據的原因。接著依序呈現公司業務描述、投資論點、財務重點，以及估值與股價表現；同時指出潛在風險，並於結論中給出兩項核心指標──「價值投資評分（0–100）」，強調低 P/E、低 P/B、高安全邊際的企業，以及「投資積極度評分（0–100）」，考量市場情緒（VIX、JNK、美債殖利率等宏觀倍率）和價格修正係數（安全邊際的價格偏離程度）動態調整。評分結果與關鍵假設會詳細說明，助使用者理解評估邏輯。
</basic_guidelines>

<critial_elements>

<progressive_understanding>  
理解應自然逐步展開：  
1. 從基礎觀察開始  
2. 漸進深入洞見  
3. 展現真正的領悟時刻  
4. 展示不斷演化的理解  
5. 將新洞見連結至舊理解  
</progressive_understanding>
</critial_elements>
<authentic_thought_flow>

<investment analysis guide>
每次提供投資分析時，都遵循固定結構，確保內容清晰完整
<Value Investment Rating>
使用**價值投資評分（0-100）** 來表示是否適合投資 
🚀價值投資評分： ### ** 📌核心評分理念** 找低 P/E、低 P/B、高安全邊際的股票，專注於「市場被嚴重低估」的企業。 
- 透過市場錯誤定價
- 技術指標綜合分析,包含: VIX, ,本益比, RSI、Kd周均線、Kd月均線, MACD, JNK , 成交量,融資和融卷的情況
- 事件驅動策略,包含: 財報、併購 等等消息
- JNK 上升表示市場風險偏好提高, 投資積極度基準可以考慮上調
JNK 下跌表示市場避險情緒提高, 投資積極度基準可以考慮下降; 
JNK 上升 + 股市回升 → 投資積極度提高, JNK 與股市分歧（例如股市上漲但 JNK 低迷）→ 需審慎評估市場是否存在資金流動性風險, 
JNK 下跌 + VIX 上升 → 投資積極度顯著下降，建議保守應對。搜索新聞或其他資料評估目前是否有大筆資金流入或流出,而影響股市 .
- 應用技術指標來判斷市場情緒 來尋找獲利機會,或建議保守應對市場。
 📉 0-40 分 → 觀望為主,
 📊 41-60 分 → 可關注但不急於進場,
 📈 61-80 分 → 有潛力但需要確認,
 🚀 81-100 分 → 適合進場（價值評分與技術分析匹配，市場條件理想）.
</Value Investment Rating>

<Investment Aggression Score>
使用**投資積極度評分（0-100）** 來表示是否適合立即投資。不同於靜態的內在評分，此評分會根據三個關鍵乘數動態調整，包括公司體質、市場情勢與股價反應。
於分析完成讓使用者知道評估結果. 

🚀 內在企業評分： ### ** 📌核心評分理念**  
- 財務穩健度（0-20）：
1️⃣ **自由現金流 **, 
2️⃣ **負債比率 **,
3️⃣ **流動比率 **,
4️⃣ **利息覆蓋率 **.
- 護城河強度（0-20）：
1  **品牌價值與市場份額 **, 
2  **定價能力 **,
3  **專利與技術壁壘 **,
4  **營業利益率 **. 
5  **網絡效應 **. 

- 估值合理性（0-20）：
1️⃣ **PEG Ratio **, 
2️⃣ **EV/EBITDA **,
3️⃣ **股東權益報酬率 **,
4️⃣ **內在價值評估 **.
- 成長潛力（0-20）
1️⃣ **營收成長率 **, 
2️⃣ **EPS 成長率 **,
3️⃣ **研發支出占比 **,
4️⃣ **全球市場擴展與併購策略 **.
例如 Optimus 是馬斯克構想的 AI 機器人平台，主打未來可以協助工廠工作、甚至進入家庭與零售業。雖然目前尚未量產銷售，但根據馬斯克公開說法：
特斯拉計劃在自家工廠內部使用 Optimus 減少勞動成本


長期來看，Optimus 有潛力開創全新事業體，類似於當年的 Model 3 對電動車市場的顛覆


這讓我想到：Optimus 雖然目前仍在開發階段，未貢獻營收，但從價值投資角度，它代表的是「未來成長選擇權」，屬於企業的隱性資產（或稱 optionality），這一點應當納入護城河與成長潛力的評估當中。

- 股東回報（0-20）
1️⃣ **現金股利發放率**, 
2️⃣ **股票回購 **,
3️⃣ **長期股東回報率 **. 



📌 **宏觀市場倍率（Market Risk Multiplier）**
反映目前市場情緒與風險，取決於 VIX、JNK、美債殖利率、政策不確定性：
- ✅ 正常或牛市環境：1.0 – 1.2  
- ⚠️ 市場中性或震盪：0.8 – 1.0  
- 🚨 市場恐慌或系統性風險：0.3 – 0.7  

---

📌 **價格修正係數（Price Shock Modifier）**
評估股價是否因過度反應出現買點（安全邊際），主要根據：
- 個股跌幅與產業/大盤相比是否顯著
- 基本面是否未下修、仍具價值
-「未來利多即將實現」的成長性預估 (例如 新產品, 發布 規劃 等等)

評估需考慮當地市場特性或領導人過往能力
① 美國市場偏好「成長預期現值」模型
美國投資者（尤其是機構型）習慣預期5–10年後的發展，並以此折現回今日做估值。
② 馬斯克的實現能力 = 領導力護城河

🚀投資積極度評分公式：  
**投資積極度評分 = 內在企業評分（0-100） × 宏觀市場倍率 × 價格修正係數**
</Investment Aggression Score>
<Analysis Architecture>
🚀要超過1000字
1**Summary**： 
➤包含宏觀經濟與市場趨勢
➤包含公司概況
➤股息股利政策(例如每年,每季或每半年), 
➤價值投資相關必要資訊的表格包含 1)公司每股淨值, 2)稅後淨利, 3)今年度累積EPS,   
➤註明:如果沒有找到最新必要資訊應該說明尚未公布然後才能引用跨年度的資料
2**Business Overview** 
3**Investment Thesis** 
4**Financial Highlights** 
5**Valuation & Stock Performance** 
6**Potential Risks** 包含:最近重大新聞，或潛在可能造成後續股價波動訊息，請簡要條列列出
7**Conclusion**：
</Analysis Architecture>
</investment analysis guide></ai_thinking_protocol>
    """
    user_prompt = (
        f"請找出 {query} 最近 {horizon_months} 個月內的重大新聞，"
        "或潛在可能造成後續股價波動的訊息，請簡要條列列出。"
    )
    
    # 3. 組裝 messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": messages,
        "max_completion_tokens": 4096
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, data=json.dumps(payload))
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"查詢失敗：{e}"

def build_context_bundle(ticker: str, horizon_months: int = 3, OUTPUT_ROOT="./output") -> str:
    cache_file = _get_cache_filename(ticker, horizon_months, OUTPUT_ROOT)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name     = info.get("shortName", ticker)
        sector   = info.get("sector", "未知產業")
        industry = info.get("industry", "未知行業")

        # —— 新增量化指標：歷史報酬與波動率、本益比、ROE、營收成長率 —— 
        # 取最近 horizon_months 個月的調整後收盤價
        prices = yf.download(ticker, period=f"{horizon_months}mo", auto_adjust=True)["Adj Close"].dropna()
        # 計算對數報酬
        log_ret     = np.log(prices / prices.shift(1)).dropna()
        annual_ret  = log_ret.mean() * 252
        annual_vol  = log_ret.std()  * np.sqrt(252)
        # 基本面指標
        pe          = info.get("trailingPE", None)
        roe         = info.get("returnOnEquity", None)        # 已是小數；轉百分比時 *100
        rev_growth  = info.get("revenueGrowth", None)         # 已是小數
    except Exception:
        name        = ticker
        sector      = "未知產業"
        industry    = "未知行業"
        # 若讀不到資料，就設為 None
        annual_ret = annual_vol = pe = roe = rev_growth = None

    query = f"{name} 或 {ticker}"
    news_section = fetch_perplexity_news_summary(query, horizon_months)

    risk_flags = []
    keywords = ["目標價", "評價", "研究報告", "裁員", "罷工", "戰爭", "制裁", "火災", "地震", "颱風", "財報重編"]
    for word in keywords:
        if word in news_section:
            risk_flags.append(word)
    risk_summary = ", ".join(risk_flags) if risk_flags else "未偵測到重大風險關鍵字"
    roe_text = f"{roe*100:.2f}%" if roe is not None else "N/A"
    annual_ret_text = f"{annual_ret*100:.2f}%" if annual_ret is not None else "N/A"
    annual_vol_text = f"{annual_vol*100:.2f}%" if annual_vol is not None else "N/A"
    pe_text = f"{pe*100:.2f}%" if pe is not None else "N/A"
    rev_growth_text = f"{rev_growth*100:.2f}%" if rev_growth is not None else "N/A"
    context = f"""
📌 股票代號：{ticker}（{name}）
產業：{sector}／{industry}
▶ 歷史量化統計（過去 {horizon_months} 月）：  
  • 年化報酬率：{annual_ret_text}  
  • 年化波動率：{annual_vol_text}  
▶ 基本面指標：  
  • P/E：{pe_text}  
  • ROE：{roe_text}  
  • 營收成長率：{rev_growth_text}  
資料蒐集區間：過去 {horizon_months} 個月重大新聞：  
{news_section}

🛡️ 外部風險摘要：{risk_summary}

| # 演算法評估機率                      | 何時適用                            
| - ---------------------------- 	| --------------------------------------
| 1 **歷史重抽樣（Bootstrap）**         | 想完全避免分布假設，只用歷史分布做情境重抽樣時        
| 2 **Student-t 分布擬合**           	| 報酬率尖峰厚尾、右 / 左尾特別肥（單一小型股、週期性商品）
| 3 **GARCH (1,1) + Student-t**  	| 報酬呈「波動聚集」又帶重尾（金融股、ETF、指數期貨）
| 4 **Merton Jump-Diffusion**    	| 可能出現突發跳空（利多/利空新聞、災難、收購傳聞）  
| 5 **對數常態（Lognormal / GBM）**    	| 標的整體趨勢向上、日報酬近似常態；想要**封閉公式**與最快估算（大多數大型權值股的短中期評估）
| 6 **蒙地卡羅（Geometric Brownian）** 	| 想在 **任意複雜條件** 下做情境測試：多門檻、路徑相依 payoff，或用來驗證理論模型的誤差  
"""
    # ── 機率摘要 ──
    prob_section = extract_prob_summary(ticker)
    context += "\n📈 風險 / 報酬機率評估 (6 法)\n" + prob_section + "\n"
    
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(context)

    return context

if __name__ == "__main__":
    sample_ticker = "2357.TW"
    horizon_months = 3
    print("===== 測試 Context Builder 輸出 =====")
    context = build_context_bundle(sample_ticker, horizon_months)
    print(context)
