#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import re
import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from datetime import datetime
RUN_DATE = datetime.today().strftime("%Y%m%d")
OUTPUT_ROOT = os.path.join("outputs", RUN_DATE)
CACHE_DIR = os.path.join(OUTPUT_ROOT, "fetch_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_00713_components():
     today = datetime.today().strftime("%Y%m%d")
     cache_file = os.path.join(CACHE_DIR, f"00713_components_{today}.json")
     # 如果當天已有快取，直接讀出並回傳
     if os.path.exists(cache_file):
         with open(cache_file, "r", encoding="utf-8") as f:
             print(f"[快取] 0050 已讀取 {cache_file}")
             return json.load(f)
     url = "https://www.yuantaetfs.com/product/detail/00713/ratio"
     headers = {
         "User-Agent": (
             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
         )
     }

     try:
         response = requests.get(url, headers=headers, timeout=10)
         response.raise_for_status()
     except requests.RequestException as e:
         print(f"⚠️ 無法下載網頁: {e}")
         return []

     html = response.text

     # 從 HTML 中找到 StockWeights 陣列
     m = re.search(r'StockWeights\s*:\s*(\[[\s\S]*?\])', html)
     if not m:
         print("⚠️ 找不到 StockWeights 資料")
         return []

     array_text = m.group(1)

     # 用正則分割出各個物件文字，並擷取 code 和 name
     objs = re.findall(r'\{([^}]+?)\}', array_text)
     print(f"找到 00713  成分股")
     components = []
     for obj in objs:
         code_m = re.search(r'code:"(\d+)"', obj)
         name_m = re.search(r'name:"([^\"]+)"', obj)
         if code_m and name_m:
             code = code_m.group(1).strip()
             name = name_m.group(1).strip()
             components.append((f"{code}.TW", name))
             print(f"{code}.TW => {name}")
     # 寫入快取檔
     with open(cache_file, "w", encoding="utf-8") as f:
         json.dump(components, f, ensure_ascii=False, indent=2)
     print(f"[快取] 已儲存 00713 成分股到 {cache_file}")
     return components


if __name__ == "__main__":
     comps = fetch_00713_components()
     print(f"\n📦 00713 成分股共 {len(comps)} 檔：")
     for code, name in comps:
         print(f"{code} => {name}")
