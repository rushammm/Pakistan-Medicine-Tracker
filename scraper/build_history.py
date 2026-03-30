"""
Build Historical Data
=====================
Generates realistic historical price records based on current real scraped data.
Uses real prices as anchors with small realistic variations over time.
This gives ML models enough data to train on.
"""

import sqlite3
import random
import numpy as np
from datetime import datetime, timedelta

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")


def build_history(weeks=8):
    """Generate historical records for the past N weeks based on real data."""
    conn = sqlite3.connect(DB_PATH)

    # Get current real records
    rows = conn.execute(
        "SELECT name, brand, generic_name, price_pkr, drap_price_pkr, source, availability "
        "FROM prices WHERE source IN ('dawaai.pk', 'dvago.pk')"
    ).fetchall()

    if not rows:
        print("No real data to build history from.")
        conn.close()
        return

    print(f"Building {weeks} weeks of history from {len(rows)} real records...")

    now = datetime.now()
    records = []

    for row in rows:
        name, brand, generic, price, drap, source, avail = row
        base_price = price

        for week in range(1, weeks + 1):
            ts = (now - timedelta(weeks=week)).isoformat()

            # Price variation: ±5% random walk from current price
            # Older records trend slightly differently to create realistic patterns
            noise = random.gauss(0, 0.03)  # 3% std dev
            trend = random.uniform(-0.01, 0.01) * week  # slight drift over time
            hist_price = round(base_price * (1 + noise + trend), 2)
            hist_price = max(1.0, hist_price)  # floor at Rs 1

            # Availability: mostly same as current, but some random changes
            avail_choices = ["In Stock", "Limited", "Out of Stock"]
            if avail == "In Stock":
                hist_avail = random.choices(avail_choices, weights=[80, 12, 8], k=1)[0]
            elif avail == "Out of Stock":
                hist_avail = random.choices(avail_choices, weights=[15, 15, 70], k=1)[0]
            else:
                hist_avail = random.choices(avail_choices, weights=[40, 40, 20], k=1)[0]

            # Overprice calc
            overprice_pct = round(((hist_price - drap) / drap) * 100, 2) if drap > 0 else 0
            overpriced = 1 if overprice_pct > 10 else 0

            records.append((
                name, brand, generic, hist_price, drap,
                overpriced, overprice_pct, source, hist_avail, ts
            ))

    # Insert all historical records
    conn.executemany("""
        INSERT INTO prices (name, brand, generic_name, price_pkr, drap_price_pkr,
                           overpriced, overprice_pct, source, availability, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()

    # Verify
    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    dates = conn.execute("SELECT COUNT(DISTINCT DATE(scraped_at)) FROM prices").fetchone()[0]
    print(f"Done! Total records: {total} across {dates} distinct dates")
    conn.close()


if __name__ == "__main__":
    build_history(weeks=8)
