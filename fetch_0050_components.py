#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def fetch_0050_components(CACHE_DIR):
    today = datetime.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"0050_components_{today}.json")
    # 如果當天已有快取，直接讀出並回傳
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            print(f"[快取] 0050 已讀取 {cache_file}")
            return json.load(f)

    url = "https://www.yuantaetfs.com/product/detail/0050/ratio"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector('text=展開', timeout=10000)
        except PlaywrightTimeout:
            print("載入頁面超時，請檢查網路或網站狀態")
            browser.close()
            return []

        # 點擊展開
        expand_locator = page.locator('text=展開')
        if expand_locator.count() > 0:
            expand_locator.first.click()
            for _ in range(10):
                rows = page.locator('div.each_table').nth(1).locator('div.tbody > div.tr')
                if rows.count() >= 50:
                    break
                page.wait_for_timeout(500)

        tables = page.locator('div.each_table')
        if tables.count() < 2:
            print("找不到完整股票列表容器")
            browser.close()
            return []
        stock_table = tables.nth(1)
        rows = stock_table.locator('div.tbody > div.tr')
        total = rows.count()
        print(f"找到 0050 {total} 列成分股")

        components = []
        for i in range(total):
            row = rows.nth(i)
            spans = row.locator('span')
            code = spans.nth(1).inner_text().strip()
            name = spans.nth(3).inner_text().strip()
            if code.isdigit():
                components.append((f"{code}.TW", name))
        browser.close()

    # 寫入快取檔
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(components, f, ensure_ascii=False, indent=2)
    print(f"[快取] 已儲存 0050 成分股到 {cache_file}")
    return components
