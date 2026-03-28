"""
Pakistan Medicine Price Scraper
===============================
Scrapes medicine prices from dawaai.pk and medstore.com.pk, compares them
against DRAP (Drug Regulatory Authority Pakistan) registered prices.

Falls back to realistic synthetic data if scraping fails.
"""

import os
import sys
import csv
import sqlite3
import random
import time
import logging
import argparse
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import schedule
from sklearn.ensemble import IsolationForest
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")
DRAP_CSV = os.path.join(BASE_DIR, "data", "drap_prices.csv")
LOG_PATH = os.path.join(BASE_DIR, "data", "scraper.log")

DAWAAI_SEARCH_URL = "https://dawaai.pk/search?q={query}"
MEDSTORE_SEARCH_URL = "https://medstore.com.pk/catalogsearch/result/?q={query}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

OVERPRICE_THRESHOLD = 10  # percentage above DRAP price to flag as overpriced

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("medicine_scraper")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(message)s"))

_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)

logger.addHandler(_console_handler)
logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create the SQLite database and prices table if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
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
    conn.commit()
    conn.close()
    logger.info("[OK] Database initialised at %s", DB_PATH)


def save_to_db(records: list[dict]):
    """Insert a list of medicine records into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for rec in records:
        cursor.execute("""
            INSERT INTO prices
                (name, brand, generic_name, price_pkr, drap_price_pkr,
                 overpriced, overprice_pct, source, availability, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["name"],
            rec.get("brand", ""),
            rec.get("generic_name", ""),
            rec["price_pkr"],
            rec.get("drap_price_pkr", 0),
            rec.get("overpriced", 0),
            rec.get("overprice_pct", 0.0),
            rec.get("source", "unknown"),
            rec.get("availability", "Unknown"),
            rec.get("scraped_at", datetime.now().isoformat()),
        ))
    conn.commit()
    conn.close()
    logger.info("[OK] Saved %d records to database", len(records))


# ---------------------------------------------------------------------------
# Load DRAP reference prices
# ---------------------------------------------------------------------------

def load_drap_prices() -> list[dict]:
    """Read the DRAP reference CSV and return a list of dicts."""
    medicines = []
    with open(DRAP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            medicines.append({
                "name": row["name"].strip(),
                "generic_name": row["generic_name"].strip(),
                "drap_price_pkr": float(row["drap_price_pkr"]),
            })
    logger.info("[OK] Loaded %d medicines from DRAP reference CSV", len(medicines))
    return medicines


# ---------------------------------------------------------------------------
# Overpricing logic
# ---------------------------------------------------------------------------

def flag_overpriced(record: dict) -> dict:
    """
    Compare scraped price against DRAP price.
    Flag as overpriced if the scraped price is >10% above DRAP.
    """
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
# Live scraper — dawaai.pk
# ---------------------------------------------------------------------------

def scrape_dawaai(medicine_name: str) -> list[dict]:
    """
    Scrape medicine prices from dawaai.pk search results.
    Returns a list of product dicts or an empty list on failure.
    """
    url = DAWAAI_SEARCH_URL.format(query=medicine_name.replace(" ", "+"))
    results = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=(3, 10))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # dawaai.pk renders product cards inside the search results page
        product_cards = soup.select("div.product-card, div.productCard, a.product-card")

        # Fallback: look for common e-commerce card patterns
        if not product_cards:
            product_cards = soup.select("[class*='product'], [class*='Product']")

        for card in product_cards[:5]:  # limit to top 5 results per medicine
            # Try to extract product name
            name_el = (
                card.select_one("h2, h3, h4, [class*='name'], [class*='title']")
            )
            name = name_el.get_text(strip=True) if name_el else None

            # Try to extract price
            price_el = card.select_one(
                "[class*='price'], [class*='Price'], span.amount"
            )
            price_text = price_el.get_text(strip=True) if price_el else None

            if name and price_text:
                # Clean price string: remove "Rs", commas, etc.
                price_clean = (
                    price_text.replace("Rs", "")
                    .replace("Rs.", "")
                    .replace(",", "")
                    .replace("PKR", "")
                    .strip()
                )
                try:
                    price_val = float(price_clean)
                except ValueError:
                    continue

                # Extract availability/stock status
                availability = "In Stock"  # default if price found
                stock_el = card.select_one(
                    "[class*='stock'], [class*='avail'], [class*='Stock'], [class*='Avail']"
                )
                if stock_el:
                    stock_text = stock_el.get_text(strip=True).lower()
                    if "out of stock" in stock_text:
                        availability = "Out of Stock"
                    elif "limited" in stock_text or "low" in stock_text:
                        availability = "Limited"

                results.append({
                    "name": name,
                    "brand": name.split()[0] if name else "",
                    "price_pkr": price_val,
                    "availability": availability,
                    "source": "dawaai.pk",
                    "scraped_at": datetime.now().isoformat(),
                })

        if results:
            logger.info("  [LIVE] Scraped %d results for '%s' from dawaai.pk", len(results), medicine_name)

    except requests.RequestException as e:
        logger.warning("  [WARN] dawaai.pk scraping failed for '%s': %s", medicine_name, e)

    return results


# ---------------------------------------------------------------------------
# Live scraper — medstore.com.pk
# ---------------------------------------------------------------------------

def scrape_medstore(medicine_name: str) -> list[dict]:
    """
    Scrape medicine prices from medstore.com.pk search results.
    Returns a list of product dicts or an empty list on failure.
    """
    url = MEDSTORE_SEARCH_URL.format(query=medicine_name.replace(" ", "+"))
    results = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=(3, 10))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # medstore.com.pk uses standard Magento-style product listing
        product_cards = soup.select("li.product-item, div.product-item, ol.products li")

        if not product_cards:
            product_cards = soup.select("[class*='product'], [class*='Product']")

        for card in product_cards[:5]:
            name_el = card.select_one(
                "a.product-item-link, h2, h3, [class*='name'], [class*='title']"
            )
            name = name_el.get_text(strip=True) if name_el else None

            price_el = card.select_one(
                "span.price, [class*='price'], [data-price-type='finalPrice']"
            )
            price_text = price_el.get_text(strip=True) if price_el else None

            if name and price_text:
                price_clean = (
                    price_text.replace("Rs", "")
                    .replace("Rs.", "")
                    .replace(",", "")
                    .replace("PKR", "")
                    .replace("₨", "")
                    .strip()
                )
                try:
                    price_val = float(price_clean)
                except ValueError:
                    continue

                # Extract availability/stock status
                availability = "In Stock"  # default if price found
                stock_el = card.select_one(
                    "[class*='stock'], [class*='avail'], [class*='Stock'], [class*='Avail']"
                )
                if stock_el:
                    stock_text = stock_el.get_text(strip=True).lower()
                    if "out of stock" in stock_text:
                        availability = "Out of Stock"
                    elif "limited" in stock_text or "low" in stock_text:
                        availability = "Limited"

                results.append({
                    "name": name,
                    "brand": name.split()[0] if name else "",
                    "price_pkr": price_val,
                    "availability": availability,
                    "source": "medstore.com.pk",
                    "scraped_at": datetime.now().isoformat(),
                })

        if results:
            logger.info("  [LIVE] Scraped %d results for '%s' from medstore.com.pk", len(results), medicine_name)

    except requests.RequestException as e:
        logger.warning("  [WARN] medstore.com.pk scraping failed for '%s': %s", medicine_name, e)

    return results


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(records: list[dict]) -> list[dict]:
    """
    Use IsolationForest to detect anomalous price points.
    Adds an 'anomaly' key (1 = anomaly, 0 = normal) to each record.
    """
    if len(records) < 5:
        for rec in records:
            rec["anomaly"] = 0
        return records

    prices = np.array([r["price_pkr"] for r in records]).reshape(-1, 1)

    # Also use overprice_pct as a feature if available
    overprice_pcts = np.array([r.get("overprice_pct", 0) for r in records]).reshape(-1, 1)
    features = np.hstack([prices, overprice_pcts])

    model = IsolationForest(contamination=0.1, random_state=42)
    predictions = model.fit_predict(features)

    for rec, pred in zip(records, predictions):
        rec["anomaly"] = 1 if pred == -1 else 0

    anomaly_count = sum(1 for r in records if r["anomaly"] == 1)
    logger.info("[OK] Anomaly detection complete: %d anomalies out of %d records", anomaly_count, len(records))

    return records


# ---------------------------------------------------------------------------
# Synthetic fallback data
# ---------------------------------------------------------------------------

def generate_synthetic(drap_medicines: list[dict]) -> list[dict]:
    """
    Generate realistic synthetic pharmacy prices based on DRAP data.
    Adds 0-40% random markup to simulate real-world pharmacy pricing.
    Also generates a few historical data points for trend charts.
    """
    records = []
    now = datetime.now()

    for med in drap_medicines:
        # Generate current price with random markup (0-40%) for dawaai.pk
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
            "source": "dawaai.pk (simulated)",
            "scraped_at": now.isoformat(),
        }
        records.append(flag_overpriced(record))

        # Generate current price for medstore.com.pk (slightly different markup)
        markup2 = random.uniform(0.0, 0.35)
        price2 = round(med["drap_price_pkr"] * (1 + markup2), 2)

        record2 = {
            "name": med["name"],
            "brand": med["name"].split()[0],
            "generic_name": med["generic_name"],
            "price_pkr": price2,
            "drap_price_pkr": med["drap_price_pkr"],
            "availability": random.choices(
                ["In Stock", "Limited", "Out of Stock"],
                weights=[75, 15, 10], k=1)[0],
            "source": "medstore.com.pk (simulated)",
            "scraped_at": now.isoformat(),
        }
        records.append(flag_overpriced(record2))

        # Generate 4 historical data points for trend charts
        for days_ago in [7, 14, 21, 30]:
            hist_markup = random.uniform(0.0, 0.40)
            hist_price = round(med["drap_price_pkr"] * (1 + hist_markup), 2)
            hist_record = {
                "name": med["name"],
                "brand": med["name"].split()[0],
                "generic_name": med["generic_name"],
                "price_pkr": hist_price,
                "drap_price_pkr": med["drap_price_pkr"],
                "availability": random.choices(
                    ["In Stock", "Limited", "Out of Stock"],
                    weights=[75, 15, 10], k=1)[0],
                "source": "dawaai.pk (simulated)",
                "scraped_at": (now - timedelta(days=days_ago)).isoformat(),
            }
            records.append(flag_overpriced(hist_record))

    logger.info("[OK] Generated %d synthetic records (with history)", len(records))
    return records


# ---------------------------------------------------------------------------
# Main scraping pipeline
# ---------------------------------------------------------------------------

def run_scraper():
    """
    Main entry point: initialise DB, attempt live scraping from both sources,
    fall back to synthetic data if needed, run anomaly detection, and save to DB.
    """
    logger.info("=" * 60)
    logger.info("  Pakistan Medicine Price Scraper")
    logger.info("=" * 60)

    # Step 1: Initialise the database
    init_db()

    # Step 2: Load DRAP reference data
    drap_medicines = load_drap_prices()

    # Step 3: Attempt live scraping from both sources
    all_records = []
    live_success = False

    logger.info("[INFO] Attempting live scraping from dawaai.pk and medstore.com.pk ...")

    # Quick reachability probe — skip a source entirely after first DNS/connection
    # failure to avoid 181 × timeout waits.
    skip_dawaai = False
    skip_medstore = False
    dawaai_conn_failures = 0
    medstore_conn_failures = 0
    _FAIL_THRESHOLD = 2  # skip source after this many consecutive connection failures

    for med in drap_medicines:
        # Scrape dawaai.pk
        if not skip_dawaai:
            scraped_dawaai = scrape_dawaai(med["name"])
            time.sleep(1)  # polite delay between requests

            for rec in scraped_dawaai:
                rec["generic_name"] = med["generic_name"]
                rec["drap_price_pkr"] = med["drap_price_pkr"]
                all_records.append(flag_overpriced(rec))

            if scraped_dawaai:
                live_success = True
                dawaai_conn_failures = 0
            else:
                dawaai_conn_failures += 1
                if dawaai_conn_failures >= _FAIL_THRESHOLD:
                    logger.info("  [SKIP] dawaai.pk unreachable after %d failures — skipping remaining", _FAIL_THRESHOLD)
                    skip_dawaai = True

        # Scrape medstore.com.pk
        if not skip_medstore:
            scraped_medstore = scrape_medstore(med["name"])
            time.sleep(1)

            for rec in scraped_medstore:
                rec["generic_name"] = med["generic_name"]
                rec["drap_price_pkr"] = med["drap_price_pkr"]
                all_records.append(flag_overpriced(rec))

            if scraped_medstore:
                live_success = True
                medstore_conn_failures = 0
            else:
                medstore_conn_failures += 1
                if medstore_conn_failures >= _FAIL_THRESHOLD:
                    logger.info("  [SKIP] medstore.com.pk unreachable after %d failures — skipping remaining", _FAIL_THRESHOLD)
                    skip_medstore = True

    # Step 4: Fall back to synthetic data if scraping returned nothing
    if not all_records:
        logger.info("[INFO] Live scraping returned no data — generating synthetic fallback ...")
        all_records = generate_synthetic(drap_medicines)
    elif not live_success:
        logger.info("[INFO] Partial scraping — supplementing with synthetic data ...")
        all_records.extend(generate_synthetic(drap_medicines))

    # Step 5: Anomaly detection
    all_records = detect_anomalies(all_records)

    # Step 6: Save everything to the database
    save_to_db(all_records)

    # Summary
    overpriced_count = sum(1 for r in all_records if r.get("overpriced"))
    anomaly_count = sum(1 for r in all_records if r.get("anomaly"))
    logger.info("=" * 60)
    logger.info("  Done! %d records saved.", len(all_records))
    logger.info("  Overpriced: %d / %d", overpriced_count, len(all_records))
    logger.info("  Anomalies detected: %d", anomaly_count)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Scheduled scraping
# ---------------------------------------------------------------------------

def run_scheduled(interval_hours: int = 6):
    """Run the scraper on a recurring schedule."""
    logger.info("[SCHEDULE] Starting scheduled scraping every %d hours", interval_hours)
    logger.info("[SCHEDULE] First run starting now...")

    run_scraper()

    schedule.every(interval_hours).hours.do(run_scraper)

    while True:
        schedule.run_pending()
        time.sleep(60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pakistan Medicine Price Scraper")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run the scraper on a recurring schedule (default: every 6 hours)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=6,
        help="Interval in hours for scheduled scraping (default: 6)",
    )
    args = parser.parse_args()

    if args.schedule:
        run_scheduled(interval_hours=args.interval)
    else:
        run_scraper()
