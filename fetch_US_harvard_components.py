#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from datetime import datetime
from urllib.parse import urlparse

import cloudscraper
from bs4 import BeautifulSoup

def fetch_US_harvard_components(cache_dir):
    """
    抓取 Harvard Endowment（由 Harvard Management Company 管理）的 13F 持倉成分股：
    回傳格式為 [ (ticker, company_name), ... ]
    """
    os.makedirs(cache_dir, exist_ok=True)
    today = datetime.today().strftime("%Y%m%d")
    cache_file = os.path.join(cache_dir, f"US_harvard_components_{today}.json")

    # 如果當天已有快取，直接讀取
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            print(f"[快取] 已讀取 {cache_file}")
            return json.load(f)

    url = "https://fintel.io/zh-hant/i/harvard-management"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Referer": "https://fintel.io/",
        "Accept-Language": "zh-TW,zh;q=0.9"
    }

    # 使用 cloudscraper 繞過 Cloudflare
    scraper = cloudscraper.create_scraper()
    try:
        resp = scraper.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"⚠️ 無法下載網頁: {e}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    components = []

    # 找出所有在成分股 table 裡的股票連結
    for a in soup.select("tbody#trans13f tr td a[href^='/zh-hant/s/us/']"):
        href = a["href"]
        raw = urlparse(href).path.rstrip("/").split("/")[-1]
        ticker = raw.upper() + ".US"
        text = a.get_text(strip=True)
        m = re.match(r'[^/]+/\s*(?P<name>.+?)[。]?$', text)
        company = m.group("name").strip() if m else text

        components.append((ticker, company))
        print(f"→ {ticker} : {company}")

    # 寫入快取
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(components, f, ensure_ascii=False, indent=2)
    print(f"[快取] 已儲存 Harvard 成分股到 {cache_file}")

    return components

if __name__ == "__main__":
    comps = fetch_US_harvard_components(cache_dir="./cache")
    print(f"\n📦 Harvard 成分股共 {len(comps)} 檔：")
    for tk, name in comps:
        print(f"{tk} => {name}")
