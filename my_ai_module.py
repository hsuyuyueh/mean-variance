from openai import OpenAI
import pandas as pd
import time
import json
import os
import re
from tqdm import tqdm
from context_builder import _get_cache_filename

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 快取檔案位置
from datetime import datetime

CACHE_FILE=""

from context_builder import build_context_bundle

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

def gpt_contextual_rating(tickers, base_mu=None, tech_indicators=None, model="gpt-4o-mini", force=False, horizon_months=3, OUTPUT_ROOT="./OUTPUT-DEFAULT"):
    BASE_DIR = OUTPUT_ROOT
    os.makedirs(BASE_DIR, exist_ok=True)
    global CACHE_FILE
    CACHE_FILE = os.path.join(BASE_DIR, "default_mu_cache.json")
    
    mu_estimates = {}
    cache = load_cache()

    # 載入預設提示
    try:
        with open("./default_prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read()
    except:
        base_prompt = "你是一位資深財經分析師，請評估下列股票的預期報酬率。"

    for tk in tqdm(tickers, desc=f"預測 {horizon_months} 個月 μ 值"):
        context = base_prompt + "\n" + build_context_bundle(tk, horizon_months, OUTPUT_ROOT)
        key = f"{tk}::{horizon_months}::{hash(context)}"

        if not force and key in cache:
            mu = cache[key]
            print(f"[快取] {tk}: {mu:.2%}")
            mu_estimates[tk] = mu
            continue
        
        # 讀取或建立 context bundle 的快取檔
        cache_file = _get_cache_filename(tk, horizon_months, OUTPUT_ROOT)
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as cf:
                context_bundle = cf.read()
        else:
            context_bundle = build_context_bundle(tk, horizon_months, OUTPUT_ROOT)



        #—— 重構：把 request/response/content 一次寫入 OUTPUT_ROOT/AI --------
        raw_dir = os.path.join(OUTPUT_ROOT, "AI")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, f"mu-{tk}.txt")
        #cache_file = _get_cache_filename(tk, horizon_months, OUTPUT_ROOT)
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as cf:
                print(f"[快取] {raw_path}")
                content = cf.read()
                # 統一 JSON 擷取流程
                pattern = r'(?s)^.*?(?=== RAW CONTENT ===)'
                content = re.sub(pattern, '', content)
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
                if json_match and json_match.group(1).strip():
                    json_str = json_match.group(1).strip()
                else:
                    obj_match = re.search(r"(\{[\s\S]*?\})", content)
                    if obj_match:
                        json_str = obj_match.group(1)
                    else:
                        raise ValueError("無法從 AI 回應中擷取到 JSON 物件")
                print(f"\n\n\n[快取] {json_str}")
                data = json.loads(json_str)
                mu_raw = float(data.get("mu_prediction", 0.0))
                mu = mu_raw / 100 if mu_raw > 1 else mu_raw
                pattern = r'(?s)^=== RAW CONTENT ===\s*```json.*?```\s*'
                content = re.sub(pattern, '', content)
                basis = content
                print(f"[說明] {tk} 的 μ = {mu} 判斷依據：{basis}")
                continue
 
        # 第一段 system: base_prompt；第二段 system: context_bundle
        messages = [
            {"role": "system", "content": f"base_prompt:{base_prompt}, 最新技術指標：{json.dumps(tech_indicators, ensure_ascii=False)}, context_bundle:{context_bundle}"},
            {"role": "user", "content": (
                "請依照以下格式回傳：\n"
                "1) JSON，包括：\n"
                "   • mu_prediction（預測 μ，數值）\n"
                "   • 技術指標數值：ma5, macd, kd_k, kd_d, year_line\n"
                "2) 不超過 2048 字的簡短趨勢解讀\n"
                "範例輸出：\n"
                "```json\n"
                "{\n"
                "  \"mu_prediction\": 0.045,\n"
                "  \"ma5\": 162.5,\n"
                "  \"macd\": 1.23,\n"
                "  \"kd_k\": 68.4,\n"
                "  \"kd_d\": 72.1,\n"
                "  \"year_line\": 169.5\n"
                "}\n"
                "分析須包含 目前已知的重大資訊足以影響大盤過個股走勢, 基本面, 資金面, 市場面, 相關必要線型情況和指數資訊。供後續用來評估3個月內的可能走勢和後續發展"
            )}
        ]

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=2048,
                temperature=1
            )
            # Debug: 印出原始回應與內容
            #print(f"DEBUG raw resp: {resp}")
            content = resp.choices[0].message.content.strip()
            #print(f"DEBUG content: {content}")


            with open(raw_path, "w", encoding="utf-8") as fw:
                # 1) RAW MESSAGES
                fw.write("=== RAW MESSAGES ===\n")
                try:
                    fw.write(json.dumps(messages, default=str, indent=2, ensure_ascii=False))
                except Exception as e:
                    fw.write(f"Error serializing messages: {e}\n{messages}")
                fw.write("\n\n")

                # 2) RAW RESPONSE OBJECT
                #fw.write("=== RAW RESPONSE OBJECT ===\n")
                #try:
                #    fw.write(json.dumps(resp, default=lambda o: str(o), indent=2, ensure_ascii=False))
                #except Exception as e:
                #    fw.write(f"Error serializing resp: {e}\n{resp}")
                #fw.write("\n\n")

                # 3) RAW CONTENT (純文字)
                fw.write("=== RAW CONTENT ===\n")
                fw.write(content)
                fw.close()
            print(f"[紀錄] 已將 raw data 寫入 {raw_path}")

            # 用正則抽出 JSON，並解析 mu_prediction 和 mu_basis
            m = re.search(r"\{[\s\S]*\}", content)
            json_str = m.group(0) if m else content
            data = json.loads(json_str)
            mu_raw = float(data.get("mu_prediction", 0.0))
            mu = mu_raw / 100 if mu_raw > 1 else mu_raw
            basis = data.get("mu_basis", "")
            # 印出 AI 說明 μ 值判斷依據
            print(f"[說明] {tk} 的 μ 判斷依據：{basis}")

        except Exception:
            mu = 0.0

        mu_estimates[tk] = mu
        cache[key] = mu
        time.sleep(1)
    print(f"[資訊] CACHE_FILE：{CACHE_FILE}")
    save_cache(cache)
    return pd.Series(mu_estimates)
# 測試
if __name__ == '__main__':
    sample = ['2330.TW', '2317.TW', '2454.TW']
    print(gpt_contextual_rating(sample))
