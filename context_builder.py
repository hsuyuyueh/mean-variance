import yfinance as yf
import requests
import os
import datetime
import json
import hashlib

PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_MODEL = "sonar"
CACHE_PATH = "./context_cache"

os.makedirs(CACHE_PATH, exist_ok=True)

def _get_cache_filename(ticker: str, horizon: int) -> str:
    key = f"{ticker}_{horizon}"
    hashed = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_PATH, f"{hashed}.json")

def fetch_perplexity_news_summary(query: str, horizon_months: int = 3) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "[錯誤] 未設定 PERPLEXITY_API_KEY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": (
                f"你是一位資深投資顧問。請整理下列公司去年整年度的經營情況與財報，若公司產業近期吸引資金流入，評估為強勢；若資金流出則調降積極度，以及過去 {horizon_months} 個月內在媒體上曝光的重大新聞 "
                "包含:董事長、CEO、總經理、法說會、記者說明會、發表會、目標價調升/調降、評價調升/調降、研究報告、財報、裁員、併購、財報爭議、罰款、戰爭影響等風險事件，並具體列出時間與事件。格式為：裁員、併購、財報爭議、罰款、戰爭影響等風險事件，並具體列出時間與事件。格式為："
                "時間＋事件摘要。若查無新聞，也請簡要說明營運重點。最後請總結重大風險是否明確可見，"
                "並加入總結段落描述：目前是否存在潛在或已發生的外部衝擊風險，或整體營運穩健。"
            )
        },
        {
            "role": "user",
            "content": f"請找出 {query} 最近 {horizon_months} 個月內的重大新聞，或潛在可能造成後續股價波動訊息，請簡要條列列出。"
        }
    ]

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": messages,
        "max_tokens": 1800,
        "temperature": 0.4
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, data=json.dumps(payload))
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"查詢失敗：{e}"

def build_context_bundle(ticker: str, horizon_months: int = 3) -> str:
    cache_file = _get_cache_filename(ticker, horizon_months)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get("shortName", ticker)
        sector = info.get("sector", "未知產業")
        industry = info.get("industry", "未知行業")
    except Exception:
        name = ticker
        sector = "未知產業"
        industry = "未知行業"

    query = f"{name} 或 {ticker}"
    news_section = fetch_perplexity_news_summary(query, horizon_months)

    risk_flags = []
    keywords = ["目標價", "評價", "研究報告", "裁員", "罷工", "戰爭", "制裁", "火災", "地震", "颱風", "財報重編"]
    for word in keywords:
        if word in news_section:
            risk_flags.append(word)
    risk_summary = ", ".join(risk_flags) if risk_flags else "未偵測到重大風險關鍵字"

    context = f"""
📌 股票代號：{ticker}（{name}）
產業：{sector}／{industry}
資料蒐集區間：過去 {horizon_months} 個月重大新聞：
{news_section}

🛡️ 外部風險摘要：{risk_summary}
"""

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(context)

    return context

if __name__ == "__main__":
    sample_ticker = "2357.TW"
    horizon_months = 3
    print("===== 測試 Context Builder 輸出 =====")
    context = build_context_bundle(sample_ticker, horizon_months)
    print(context)
