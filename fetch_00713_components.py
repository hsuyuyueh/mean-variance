#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# 參數與目錄設定
RUN_DATE = datetime.today().strftime("%Y%m%d")
OUTPUT_ROOT = "./outputs"
DEF_CACHE_DIR = os.path.join(OUTPUT_ROOT, "fetch_cache")
os.makedirs(DEF_CACHE_DIR, exist_ok=True)

def fetch_00713_components(CACHE_DIR=DEF_CACHE_DIR):
    today = datetime.today().strftime("%Y%m%d")
    ccache_inputfile = os.path.join("./inputs/TW/fetch_cache", f"00713_components.json")
    os.makedirs("./inputs/TW/fetch_cache", exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"00713_components_{today}.json")
    # 如果當天已有快取，直接讀出並回傳
    if os.path.exists(ccache_inputfile):
        print(f"[快取] 0056 已讀取 {cache_file}")
        with open(ccache_inputfile, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://www.yuantaetfs.com/product/detail/00713/ratio"
    components = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        # 等待 Nuxt 狀態注入完成
        page.wait_for_load_state("networkidle")
        
        # 直接從 Nuxt 全域物件抓出 FundWeights.StockWeights
        stock_weights = page.evaluate("""
            () => {
                // 找到有 weightData.FundWeights 的那個組件
                const pageData = window.__NUXT__.data.find(
                    d => d.weightData && d.weightData.FundWeights
                );
                return pageData
                    ? pageData.weightData.FundWeights.StockWeights
                    : [];
            }
        """)
        # 解析成 (code, name) 並加上 .TW 後綴
        for item in stock_weights:
            code = item["code"].strip()
            name = item["name"].strip()
            components.append((f"{code}.TW", name))
        
        print(f"[JSON] 已解析到 {len(components)} 檔股票成分")
        browser.close()

    # 寫入快取檔
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(components, f, ensure_ascii=False, indent=2)
    if not os.path.exists(ccache_inputfile):
        os.makedirs("./inputs/TW/fetch_cache", exist_ok=True)
        with open(ccache_inputfile, "w", encoding="utf-8") as f:
            json.dump(components, f, ensure_ascii=False, indent=2)

    print(f"[快取] 已儲存 00713 成分股到 {cache_file}")
    return components

if __name__ == "__main__":
     comps = fetch_00713_components()
     print(f"\n📦 00713 成分股共 {len(comps)} 檔：")
     for code, name in comps:
         print(f"{code} => {name}")
