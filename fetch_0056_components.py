#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright

from datetime import datetime
RUN_DATE = datetime.today().strftime("%Y%m%d")
OUTPUT_ROOT = os.path.join("outputs", RUN_DATE)
CACHE_DIR = os.path.join(OUTPUT_ROOT, "fetch_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_0056_components():
    today = datetime.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"0056_components_{today}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            print(f"[快取] 0056 已讀取 {cache_file}")
            return json.load(f)

    url = "https://www.yuantaetfs.com/product/detail/0056/ratio"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

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
        print(f"找到 0056 {total} 列成分股")

        components = []
        for i in range(total):
            row = rows.nth(i)
            spans = row.locator('span')
            code = spans.nth(1).inner_text().strip()
            name = spans.nth(3).inner_text().strip()
            if code.isdigit():
                components.append((f"{code}.TW", name))
        browser.close()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(components, f, ensure_ascii=False, indent=2)
    print(f"[快取] 已儲存 0056 成分股到 {cache_file}")
    return components
