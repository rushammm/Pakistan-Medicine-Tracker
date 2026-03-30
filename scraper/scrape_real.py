"""
Real Data Scraper — dawaai.pk
=============================
Scrapes medicine prices from dawaai.pk using their server-rendered
JSON-LD structured data. No Playwright needed.

Strategy:
1. Fetch sitemap to find medicine page URLs
2. Match against our DRAP medicine list
3. Scrape JSON-LD product data from each page
4. Compare against DRAP prices and save to DB

Usage:
    python scraper/scrape_real.py              # One-time scrape
    python scraper/scrape_real.py --schedule   # Run every 6 hours
"""

import os
import sys
import csv
import re
import json
import sqlite3
import time
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")
DRAP_CSV = os.path.join(BASE_DIR, "data", "drap_prices.csv")
LOG_PATH = os.path.join(BASE_DIR, "data", "scraper.log")
DAWAAI_SITEMAP_URL = "https://dawaai.pk/sitemap.xml"
DVAGO_SITEMAP_URL = "https://www.dvago.pk/product-sitemap.xml"
DVAGO_API_URL = "https://apidb.dvago.pk/AppAPIV3/GetProductDetailBySlugV2&ProductSlug={slug}&BranchCode=32"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

OVERPRICE_THRESHOLD = 10
MAX_WORKERS = 5  # Concurrent requests (be polite)
REQUEST_DELAY = 0.5  # Seconds between batches

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("real_scraper")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))

_fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(_ch)
logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            brand         TEXT,
            generic_name  TEXT,
            price_pkr     REAL NOT NULL,
            drap_price_pkr REAL,
            overpriced    INTEGER DEFAULT 0,
            overprice_pct REAL DEFAULT 0.0,
            source        TEXT,
            availability  TEXT DEFAULT 'Unknown',
            scraped_at    TEXT NOT NULL
        )
    """)
    # Scrape log table for transparency
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scrape_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scraped_at  TEXT NOT NULL,
            total_attempted INTEGER,
            total_success   INTEGER,
            total_failed    INTEGER,
            source      TEXT,
            is_real_data INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


def save_to_db(records):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for rec in records:
        cursor.execute("""
            INSERT INTO prices
                (name, brand, generic_name, price_pkr, drap_price_pkr,
                 overpriced, overprice_pct, source, availability, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["name"], rec.get("brand", ""), rec.get("generic_name", ""),
            rec["price_pkr"], rec.get("drap_price_pkr", 0),
            rec.get("overpriced", 0), rec.get("overprice_pct", 0.0),
            rec.get("source", "unknown"), rec.get("availability", "Unknown"),
            rec.get("scraped_at", datetime.now().isoformat()),
        ))
    conn.commit()
    conn.close()
    logger.info("[OK] Saved %d records to database", len(records))


def log_scrape_run(total_attempted, total_success, total_failed, source, is_real):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO scrape_log (scraped_at, total_attempted, total_success, total_failed, source, is_real_data)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), total_attempted, total_success, total_failed, source, int(is_real)))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# DRAP Reference
# ---------------------------------------------------------------------------

def load_drap_prices():
    medicines = []
    with open(DRAP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            medicines.append({
                "name": row["name"].strip(),
                "generic_name": row["generic_name"].strip(),
                "drap_price_pkr": float(row["drap_price_pkr"]),
            })
    logger.info("[OK] Loaded %d medicines from DRAP CSV", len(medicines))
    return medicines


# ---------------------------------------------------------------------------
# Overpricing
# ---------------------------------------------------------------------------

def flag_overpriced(record):
    drap = record.get("drap_price_pkr", 0)
    price = record.get("price_pkr", 0)
    if drap and drap > 0:
        pct = ((price - drap) / drap) * 100
        record["overprice_pct"] = round(pct, 2)
        record["overpriced"] = 1 if pct > OVERPRICE_THRESHOLD else 0
    else:
        record["overprice_pct"] = 0.0
        record["overpriced"] = 0
    return record


# ---------------------------------------------------------------------------
# Sitemap Fetching & URL Matching
# ---------------------------------------------------------------------------

def fetch_dawaai_urls():
    """Fetch all medicine page URLs from dawaai.pk sitemap."""
    logger.info("[INFO] Fetching sitemap from dawaai.pk...")
    try:
        r = requests.get(DAWAAI_SITEMAP_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        urls = re.findall(r'<loc>(https://dawaai\.pk/medicine/[^<]+)</loc>', r.text)
        logger.info("[OK] Found %d medicine URLs in dawaai.pk sitemap", len(urls))
        return urls
    except Exception as e:
        logger.error("[FAIL] Could not fetch dawaai sitemap: %s", e)
        return []


def fetch_dvago_slugs():
    """Fetch all product slugs from dvago.pk sitemap."""
    logger.info("[INFO] Fetching sitemap from dvago.pk...")
    try:
        slugs = []
        for sitemap_url in [DVAGO_SITEMAP_URL, DVAGO_SITEMAP_URL.replace('.xml', '-2.xml')]:
            r = requests.get(sitemap_url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                found = re.findall(r'<loc>https://www\.dvago\.pk/p/([^<]+)</loc>', r.text)
                slugs.extend(found)
        logger.info("[OK] Found %d product slugs in dvago.pk sitemap", len(slugs))
        return slugs
    except Exception as e:
        logger.error("[FAIL] Could not fetch dvago sitemap: %s", e)
        return []


def match_drap_to_dawaai(drap_medicines, sitemap_urls):
    """Match each DRAP medicine to the best dawaai.pk URL."""
    url_index = {}
    for url in sitemap_urls:
        path = url.split("/medicine/")[-1].split(".html")[0].lower()
        path_clean = re.sub(r'-\d+$', '', path)
        url_index[url] = path_clean

    matches = {}
    for med in drap_medicines:
        med_name = med["name"].lower()
        # Try multiple matching strategies
        first_word = med_name.split()[0]
        # Also try first two words for specificity (e.g. "panadol extra")
        two_words = re.sub(r'[^a-z]', '', ''.join(med_name.split()[:2]))

        best_url = None
        best_score = 0

        for url, path in url_index.items():
            # Must contain first word
            if first_word not in path:
                continue

            med_norm = re.sub(r'[^a-z0-9]', '', med_name)
            path_norm = re.sub(r'[^a-z0-9]', '', path)

            # Bonus for two-word match
            score = SequenceMatcher(None, med_norm, path_norm).ratio()
            if two_words in path_norm:
                score += 0.15

            if score > best_score:
                best_score = score
                best_url = url

        if best_url and best_score > 0.25:
            matches[med["name"]] = {
                "url": best_url,
                "score": best_score,
                "drap": med,
            }

    logger.info("[OK] Matched %d / %d DRAP medicines to dawaai.pk URLs",
                len(matches), len(drap_medicines))
    return matches


def match_drap_to_dvago(drap_medicines, dvago_slugs):
    """Match each DRAP medicine to the best dvago.pk product slug."""
    matches = {}
    for med in drap_medicines:
        med_name = med["name"].lower()
        first_word = med_name.split()[0]
        two_words = re.sub(r'[^a-z]', '', ''.join(med_name.split()[:2]))

        best_slug = None
        best_score = 0

        for slug in dvago_slugs:
            slug_lower = slug.lower()
            if first_word not in slug_lower:
                continue

            med_norm = re.sub(r'[^a-z0-9]', '', med_name)
            slug_norm = re.sub(r'[^a-z0-9]', '', slug_lower)

            score = SequenceMatcher(None, med_norm, slug_norm).ratio()
            if two_words in slug_norm:
                score += 0.15

            if score > best_score:
                best_score = score
                best_slug = slug

        if best_slug and best_score > 0.25:
            matches[med["name"]] = {
                "slug": best_slug,
                "score": best_score,
                "drap": med,
            }

    logger.info("[OK] Matched %d / %d DRAP medicines to dvago.pk slugs",
                len(matches), len(drap_medicines))
    return matches


# ---------------------------------------------------------------------------
# Page Scraping
# ---------------------------------------------------------------------------

def scrape_medicine_page(url):
    """Scrape a single dawaai.pk medicine page for JSON-LD product data."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        # Find Product JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            text = script.string or ""
            if '"Product"' not in text:
                continue

            # Clean control characters
            text = re.sub(r'[\x00-\x1f]', ' ', text)
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue

            if data.get("@type") != "Product":
                continue

            offers = data.get("offers", {})
            price_str = offers.get("price", "0")
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                continue

            if price <= 0:
                continue

            availability_url = offers.get("availability", "")
            if "InStock" in availability_url:
                availability = "In Stock"
            elif "OutOfStock" in availability_url:
                availability = "Out of Stock"
            elif "LimitedAvailability" in availability_url:
                availability = "Limited"
            else:
                availability = "Unknown"

            brand_info = data.get("brand", {})
            brand = brand_info.get("name", "") if isinstance(brand_info, dict) else ""

            return {
                "product_name": data.get("name", ""),
                "brand": brand,
                "price": price,
                "availability": availability,
                "url": url,
                "sku": data.get("sku", ""),
            }

        return None

    except requests.RequestException as e:
        logger.debug("[FAIL] %s: %s", url[-50:], e)
        return None


# ---------------------------------------------------------------------------
# dvago.pk Scraping (JSON API)
# ---------------------------------------------------------------------------

def scrape_dvago_product(slug):
    """Scrape a single dvago.pk product via their API."""
    try:
        url = DVAGO_API_URL.format(slug=slug)
        dvago_headers = {**HEADERS, "Accept": "application/json",
                         "Origin": "https://www.dvago.pk",
                         "Referer": "https://www.dvago.pk/"}
        r = requests.get(url, headers=dvago_headers, timeout=10)
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return None

        data = json.loads(r.text)
        items = data.get("Data", [])
        if not items:
            return None

        item = items[0] if isinstance(items, list) else items
        price = float(item.get("SalePrice", item.get("Price", 0)))
        if price <= 0:
            return None

        return {
            "product_name": item.get("Title", ""),
            "brand": "",
            "price": price,
            "availability": "In Stock" if price > 0 else "Unknown",
            "slug": slug,
        }
    except Exception as e:
        logger.debug("[FAIL] dvago %s: %s", slug[:40], e)
        return None


# ---------------------------------------------------------------------------
# Batch Scraping with Concurrency
# ---------------------------------------------------------------------------

def scrape_all_dvago(matches):
    """Scrape all matched medicines from dvago.pk API."""
    results = []
    total = len(matches)
    success = 0
    failed = 0

    logger.info("[INFO] Scraping %d products from dvago.pk API...", total)
    items = list(matches.items())

    for i in range(0, len(items), MAX_WORKERS):
        batch = items[i:i + MAX_WORKERS]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for med_name, info in batch:
                future = executor.submit(scrape_dvago_product, info["slug"])
                futures[future] = (med_name, info)

            for future in as_completed(futures):
                med_name, info = futures[future]
                drap = info["drap"]
                scraped = future.result()

                if scraped and scraped["price"] > 0:
                    record = {
                        "name": med_name,
                        "brand": scraped["brand"],
                        "generic_name": drap["generic_name"],
                        "price_pkr": scraped["price"],
                        "drap_price_pkr": drap["drap_price_pkr"],
                        "availability": scraped["availability"],
                        "source": "dvago.pk",
                        "scraped_at": datetime.now().isoformat(),
                    }
                    results.append(flag_overpriced(record))
                    success += 1
                else:
                    failed += 1

        done = min(i + MAX_WORKERS, total)
        if done % 20 == 0 or done == total:
            logger.info("  [%d/%d] dvago scraped (%d success, %d failed)",
                         done, total, success, failed)

        if i + MAX_WORKERS < len(items):
            time.sleep(REQUEST_DELAY)

    return results, success, failed


def scrape_all_medicines(matches):
    """Scrape all matched medicines concurrently."""
    results = []
    total = len(matches)
    success = 0
    failed = 0

    logger.info("[INFO] Scraping %d medicine pages from dawaai.pk...", total)

    items = list(matches.items())

    # Process in batches to be polite
    batch_size = MAX_WORKERS
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for med_name, info in batch:
                future = executor.submit(scrape_medicine_page, info["url"])
                futures[future] = (med_name, info)

            for future in as_completed(futures):
                med_name, info = futures[future]
                drap = info["drap"]
                scraped = future.result()

                if scraped and scraped["price"] > 0:
                    record = {
                        "name": med_name,
                        "brand": scraped["brand"],
                        "generic_name": drap["generic_name"],
                        "price_pkr": scraped["price"],
                        "drap_price_pkr": drap["drap_price_pkr"],
                        "availability": scraped["availability"],
                        "source": "dawaai.pk",
                        "scraped_at": datetime.now().isoformat(),
                    }
                    results.append(flag_overpriced(record))
                    success += 1
                else:
                    failed += 1

        # Progress
        done = min(i + batch_size, total)
        logger.info("  [%d/%d] scraped (%d success, %d failed)",
                     done, total, success, failed)

        # Polite delay between batches
        if i + batch_size < len(items):
            time.sleep(REQUEST_DELAY)

    return results, success, failed


# ---------------------------------------------------------------------------
# Anomaly Detection
# ---------------------------------------------------------------------------

def detect_anomalies(records):
    if len(records) < 5:
        for r in records:
            r["anomaly"] = 0
        return records

    prices = np.array([r["price_pkr"] for r in records]).reshape(-1, 1)
    overprice = np.array([r.get("overprice_pct", 0) for r in records]).reshape(-1, 1)
    features = np.hstack([prices, overprice])

    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(features)

    for rec, pred in zip(records, preds):
        rec["anomaly"] = 1 if pred == -1 else 0

    anomaly_count = sum(1 for r in records if r["anomaly"] == 1)
    logger.info("[OK] Anomaly detection: %d anomalies / %d records", anomaly_count, len(records))
    return records


# ---------------------------------------------------------------------------
# Synthetic Fallback (only for unmatched medicines)
# ---------------------------------------------------------------------------

def generate_synthetic_for_missing(drap_medicines, scraped_names):
    """Generate synthetic data ONLY for medicines we couldn't scrape."""
    import random
    records = []
    now = datetime.now()
    scraped_lower = {n.lower() for n in scraped_names}

    for med in drap_medicines:
        if med["name"].lower() in scraped_lower:
            continue  # Already have real data

        markup = random.uniform(0.0, 0.40)
        price = round(med["drap_price_pkr"] * (1 + markup), 2)

        record = {
            "name": med["name"],
            "brand": med["name"].split()[0],
            "generic_name": med["generic_name"],
            "price_pkr": price,
            "drap_price_pkr": med["drap_price_pkr"],
            "availability": random.choices(
                ["In Stock", "Limited", "Out of Stock"],
                weights=[75, 15, 10], k=1)[0],
            "source": "estimated",
            "scraped_at": now.isoformat(),
        }
        records.append(flag_overpriced(record))

    if records:
        logger.info("[OK] Generated %d estimated records for unmatched medicines", len(records))
    return records


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_scraper():
    logger.info("=" * 60)
    logger.info("  Real Data Scraper — dawaai.pk")
    logger.info("=" * 60)

    init_db()
    drap_medicines = load_drap_prices()
    all_records = []

    # ---- SOURCE 1: dawaai.pk ----
    dawaai_urls = fetch_dawaai_urls()
    dawaai_success = dawaai_failed = 0
    if dawaai_urls:
        dawaai_matches = match_drap_to_dawaai(drap_medicines, dawaai_urls)
        dawaai_records, dawaai_success, dawaai_failed = scrape_all_medicines(dawaai_matches)
        all_records.extend(dawaai_records)
        log_scrape_run(len(dawaai_matches), dawaai_success, dawaai_failed, "dawaai.pk", True)

    # ---- SOURCE 2: dvago.pk ----
    dvago_slugs = fetch_dvago_slugs()
    dvago_success = dvago_failed = 0
    if dvago_slugs:
        dvago_matches = match_drap_to_dvago(drap_medicines, dvago_slugs)
        dvago_records, dvago_success, dvago_failed = scrape_all_dvago(dvago_matches)
        all_records.extend(dvago_records)
        log_scrape_run(len(dvago_matches), dvago_success, dvago_failed, "dvago.pk", True)

    # ---- Fill gaps with estimated data ----
    scraped_names = set(r["name"] for r in all_records)
    estimated = generate_synthetic_for_missing(drap_medicines, scraped_names)
    all_records.extend(estimated)

    # ---- Anomaly detection ----
    all_records = detect_anomalies(all_records)

    # ---- Save ----
    save_to_db(all_records)

    # ---- Summary ----
    dawaai_count = sum(1 for r in all_records if r["source"] == "dawaai.pk")
    dvago_count = sum(1 for r in all_records if r["source"] == "dvago.pk")
    estimated_count = sum(1 for r in all_records if r["source"] == "estimated")
    overpriced = sum(1 for r in all_records if r.get("overpriced"))

    logger.info("=" * 60)
    logger.info("  Done! %d total records saved", len(all_records))
    logger.info("  dawaai.pk:  %d records (%d/%d success)",
                dawaai_count, dawaai_success, dawaai_success + dawaai_failed)
    logger.info("  dvago.pk:   %d records (%d/%d success)",
                dvago_count, dvago_success, dvago_success + dvago_failed)
    logger.info("  estimated:  %d records", estimated_count)
    logger.info("  Overpriced: %d", overpriced)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real medicine price scraper")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule")
    parser.add_argument("--interval", type=int, default=6, help="Hours between runs")
    args = parser.parse_args()

    if args.schedule:
        import schedule as sched
        logger.info("[SCHEDULE] Running every %d hours", args.interval)
        run_scraper()
        sched.every(args.interval).hours.do(run_scraper)
        while True:
            sched.run_pending()
            time.sleep(60)
    else:
        run_scraper()
