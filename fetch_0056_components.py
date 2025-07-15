#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from playwright.sync_api import sync_playwright


def fetch_0056_components():
    url = "https://www.yuantaetfs.com/product/detail/0056/ratio"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        
        # 1. 點擊「展開」按鈕，確保載入全部資料
        expand_locator = page.locator('text=展開')
        if expand_locator.count() > 0:
            expand_locator.first.click()
            # 等待列表從 5 筆展開到 50 筆
            for _ in range(10):
                rows = page.locator('div.each_table').nth(1).locator('div.tbody > div.tr')
                count = rows.count()
                if count >= 50:
                    break
                page.wait_for_timeout(500)
        
        # 2. 定位到第二個 each_table (股票列表)
        tables = page.locator('div.each_table')
        if tables.count() < 2:
            print("找不到完整股票列表容器")
            browser.close()
            return []
        stock_table = tables.nth(1)
        
        # 3. 讀取所有列
        rows = stock_table.locator('div.tbody > div.tr')
        total = rows.count()
        print(f"找到 {total} 列成分股")
        
        components = []
        for i in range(total):
            row = rows.nth(i)
            spans = row.locator('span')
            # spans[1] 是代碼, spans[3] 是名稱
            code = spans.nth(1).inner_text().strip()
            name = spans.nth(3).inner_text().strip()
            if code.isdigit():
                components.append((f"{code}.TW", name))
        
        browser.close()
        return components

if __name__ == "__main__":
    comps = fetch_0056_components()
    print(f"0056 成分股共 {len(comps)} 檔：")
    for code, name in comps:
        print(f"{code} => {name}")
