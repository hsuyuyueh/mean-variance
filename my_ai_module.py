from openai import OpenAI
import pandas as pd
import time
import json
import os
from tqdm import tqdm
from context_builder import build_context_bundle

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

mu_schema = {
    "name": "report_mu",
    "description": "預測指定股票的預期報酬率（未來一段期間內）",
    "parameters": {
        "type": "object",
        "properties": {
            "mu_prediction": {
                "type": "number",
                "description": "股票未來一段期間內預期報酬率，小數格式，例如 0.02 代表 2%"
            }
        },
        "required": ["mu_prediction"]
    }
}

CACHE_FILE = "./default_mu_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print("⚠️ 快取檔案為空，將略過使用")
                return {}
            try:
                return json.loads(content)
            except Exception as e:
                print(f"⚠️ 快取檔案格式錯誤：{e}")
                return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        with open("./default_mu_cache.json", "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

def gpt_contextual_rating(tickers, model="gpt-3.5-turbo-1106", force=False, horizon_months=3):
    mu_estimates = {}
    cache = load_cache()

    try:
        with open("./default_prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read()
    except Exception as e:
        print("讀取 default_prompt.txt 失敗：", e)
        base_prompt = "你是一位財經分析師，請評估下列股票的預期報酬率。"

    for tk in tqdm(tickers, desc=f"預測 {horizon_months} 個月 μ 值"):
        context = base_prompt + "\n" + build_context_bundle(tk, horizon_months)
        cache_key = f"{model}::{tk}::{horizon_months}::{hash(context)}"

        if not force and cache_key in cache:
            mu_estimates[tk] = cache[cache_key]
            print(f"[快取] {tk}: {cache[cache_key]:.2%}")
            continue

        messages = [
            {"role": "system", "content": context}
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[{"type": "function", "function": mu_schema}],
                tool_choice={"type": "function", "function": {"name": "report_mu"}},
                temperature=0.4,
                max_tokens=100
            )
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            parsed = json.loads(arguments)
            mu = float(parsed["mu_prediction"])
            #mu = max(0.0, min(mu, 0.2))  # 限制在 0% 到 20% 之間
            mu_estimates[tk] = mu
            cache[cache_key] = mu
            print(f"[GPT] {tk}: {mu:.2%}")
        except Exception as e:
            print(f"[錯誤] {tk}: GPT 回應解析失敗: {e}")
            mu_estimates[tk] = 0.05
            cache[cache_key] = 0.05

        time.sleep(1.2)

    save_cache(cache)
    return pd.Series(mu_estimates)

if __name__ == '__main__':
    sample = ['2357.TW', '2330.TW', '2317.TW']
    mu = gpt_contextual_rating(sample)
    print(mu)
