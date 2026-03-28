"""
Pakistan Medicine Price Scraper
===============================
Scrapes medicine prices from dawaai.pk and compares them against
DRAP (Drug Regulatory Authority Pakistan) registered prices.

Falls back to realistic synthetic data if scraping fails.
"""

import os
import sys
import csv
import sqlite3
import random
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")
DRAP_CSV = os.path.join(BASE_DIR, "data", "drap_prices.csv")

SEARCH_URL = "https://dawaai.pk/search?q={query}"
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
            scraped_at    TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[OK] Database initialised at {DB_PATH}")


def save_to_db(records: list[dict]):
    """Insert a list of medicine records into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for rec in records:
        cursor.execute("""
            INSERT INTO prices
                (name, brand, generic_name, price_pkr, drap_price_pkr,
                 overpriced, overprice_pct, source, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["name"],
            rec.get("brand", ""),
            rec.get("generic_name", ""),
            rec["price_pkr"],
            rec.get("drap_price_pkr", 0),
            rec.get("overpriced", 0),
            rec.get("overprice_pct", 0.0),
            rec.get("source", "unknown"),
            rec.get("scraped_at", datetime.now().isoformat()),
        ))
    conn.commit()
    conn.close()
    print(f"[OK] Saved {len(records)} records to database")


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
    print(f"[OK] Loaded {len(medicines)} medicines from DRAP reference CSV")
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
    url = SEARCH_URL.format(query=medicine_name.replace(" ", "+"))
    results = []

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
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

                results.append({
                    "name": name,
                    "brand": name.split()[0] if name else "",
                    "price_pkr": price_val,
                    "source": "dawaai.pk",
                    "scraped_at": datetime.now().isoformat(),
                })

        if results:
            print(f"  [LIVE] Scraped {len(results)} results for '{medicine_name}'")

    except requests.RequestException as e:
        print(f"  [WARN] Scraping failed for '{medicine_name}': {e}")

    return results


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
        # Generate current price with random markup (0-40%)
        markup = random.uniform(0.0, 0.40)
        price = round(med["drap_price_pkr"] * (1 + markup), 2)

        record = {
            "name": med["name"],
            "brand": med["name"].split()[0],
            "generic_name": med["generic_name"],
            "price_pkr": price,
            "drap_price_pkr": med["drap_price_pkr"],
            "source": "dawaai.pk (simulated)",
            "scraped_at": now.isoformat(),
        }
        records.append(flag_overpriced(record))

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
                "source": "dawaai.pk (simulated)",
                "scraped_at": (now - timedelta(days=days_ago)).isoformat(),
            }
            records.append(flag_overpriced(hist_record))

    print(f"[OK] Generated {len(records)} synthetic records (with history)")
    return records


# ---------------------------------------------------------------------------
# Main scraping pipeline
# ---------------------------------------------------------------------------

def run_scraper():
    """
    Main entry point: initialise DB, attempt live scraping, fall back to
    synthetic data if needed, flag overpriced medicines, and save to DB.
    """
    print("=" * 60)
    print("  Pakistan Medicine Price Scraper")
    print("=" * 60)
    print()

    # Step 1: Initialise the database
    init_db()

    # Step 2: Load DRAP reference data
    drap_medicines = load_drap_prices()

    # Step 3: Attempt live scraping from dawaai.pk
    all_records = []
    live_success = False

    print("\n[INFO] Attempting live scraping from dawaai.pk ...")
    for med in drap_medicines:
        scraped = scrape_dawaai(med["name"])
        time.sleep(1)  # polite delay between requests

        for rec in scraped:
            rec["generic_name"] = med["generic_name"]
            rec["drap_price_pkr"] = med["drap_price_pkr"]
            all_records.append(flag_overpriced(rec))

        if scraped:
            live_success = True

    # Step 4: Fall back to synthetic data if scraping returned nothing
    if not all_records:
        print("\n[INFO] Live scraping returned no data — generating synthetic fallback ...")
        all_records = generate_synthetic(drap_medicines)
    elif not live_success:
        print("\n[INFO] Partial scraping — supplementing with synthetic data ...")
        all_records.extend(generate_synthetic(drap_medicines))

    # Step 5: Save everything to the database
    save_to_db(all_records)

    # Summary
    overpriced_count = sum(1 for r in all_records if r.get("overpriced"))
    print(f"\n{'=' * 60}")
    print(f"  Done! {len(all_records)} records saved.")
    print(f"  Overpriced: {overpriced_count} / {len(all_records)}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_scraper()
