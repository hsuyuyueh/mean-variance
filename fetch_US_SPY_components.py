#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import requests
import re
import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

from datetime import datetime
#RUN_DATE = datetime.today().strftime("%Y%m%d")
#OUTPUT_ROOT = os.getenv('OUTPUT_ROOT')
#CACHE_DIR = os.path.join(OUTPUT_ROOT, "fetch_cache")
#os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_DIR=""

def fetch_US_SPY_components(CACHE_DIR):
     os.makedirs(CACHE_DIR, exist_ok=True)
     today = datetime.today().strftime("%Y%m%d")
     cache_file = os.path.join(CACHE_DIR, f"US_SPY_components_{today}.json")
     # 如果當天已有快取，直接讀出並回傳
     if os.path.exists(cache_file):
         with open(cache_file, "r", encoding="utf-8") as f:
             print(f"[快取] US_SPY_components_ 已讀取 {cache_file}")
             return json.load(f)
     url = "https://www.moneydj.com/ETF/X/Basic/Basic0007B.xdjhtm?etfid=SPY"
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

     # 直接解析 HTML table 中 <td class="col05"> 的 <a> 標籤
     soup = BeautifulSoup(html, "lxml")
     components = []
     # 選出所有 class=col05 且 href 包含 etfid 的 <a>
     for a in soup.select("td.col05 a[href*='etfid=']"):
         href = a["href"]
         # 拆出 URL 中的 etfid 參數作為 ticker
         qs = parse_qs(urlparse(href).query)
         ticker = qs.get("etfid", [""])[0].strip()
         name = a.get_text(strip=True)
         if ticker and name:
             components.append((ticker, name))
             print(f"{ticker} => {name}")
     # 寫入快取檔
     with open(cache_file, "w", encoding="utf-8") as f:
         json.dump(components, f, ensure_ascii=False, indent=2)
     print(f"[快取] 已儲存 US_SPY 成分股到 {cache_file}")
     return components


if __name__ == "__main__":
     comps = fetch_US_SPY_components()
     print(f"\n📦 US_SPY 成分股共 {len(comps)} 檔：")
     for code, name in comps:
         print(f"{code} => {name}")
