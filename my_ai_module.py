from openai import OpenAI
import pandas as pd
import time
import json
import os
import re
from tqdm import tqdm
from context_builder import build_context_bundle

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 快取檔案位置
from datetime import datetime
RUN_DATE = datetime.today().strftime('%Y%m%d')
BASE_DIR = os.path.join("./outputs", RUN_DATE)
os.makedirs(BASE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(BASE_DIR, "default_mu_cache.json")


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# 主函式：預測 μ 並 debug 原始資料，處理 markdown fence

def gpt_contextual_rating(tickers, model="gpt-4o-mini", force=False, horizon_months=3):
    mu_estimates = {}
    cache = load_cache()

    # 載入預設提示
    try:
        with open("./default_prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read()
    except:
        base_prompt = "你是一位資深財經分析師，請評估下列股票的預期報酬率。"

    for tk in tqdm(tickers, desc=f"預測 {horizon_months} 個月 μ 值"):
        context = base_prompt + "\n" + build_context_bundle(tk, horizon_months)
        key = f"{tk}::{horizon_months}::{hash(context)}"

        if not force and key in cache:
            mu = cache[key]
            print(f"[快取] {tk}: {mu:.2%}")
            mu_estimates[tk] = mu
            continue

        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": (
                "請僅回傳純 JSON {\"mu_prediction\": <數值>}，勿加其他文字。"
            )}
        ]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=100,
                temperature=1
            )
            # Debug: 印出原始回應與內容
            print(f"DEBUG raw resp: {resp}")
            content = resp.choices[0].message.content.strip()
            print(f"DEBUG content: {content}")

            # 用正則抽出最內層 JSON
            m = re.search(r"\{.*?\}", content, flags=re.S)
            json_str = m.group(0) if m else content
            data = json.loads(json_str)
            mu_raw = float(data.get("mu_prediction", 0.0))
            # 若回傳 >1，視為百分比，轉換小數
            mu = mu_raw / 100 if mu_raw > 1 else mu_raw
        except Exception:
            mu = 0.0

        mu_estimates[tk] = mu
        cache[key] = mu
        time.sleep(1)

    save_cache(cache)
    return pd.Series(mu_estimates)
# 測試
if __name__ == '__main__':
    sample = ['2330.TW', '2317.TW', '2454.TW']
    print(gpt_contextual_rating(sample))
