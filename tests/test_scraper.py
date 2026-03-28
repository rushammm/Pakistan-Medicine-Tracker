"""
Unit tests for the Pakistan Medicine Price Scraper.
Run with: python -m pytest tests/
"""

import os
import csv
import sqlite3
import tempfile

import pytest

# Ensure project root is on path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHARMACIES_CSV = os.path.join(BASE_DIR, "data", "pharmacies.csv")

from scraper.scrape import (
    flag_overpriced,
    load_drap_prices,
    generate_synthetic,
    init_db,
    detect_anomalies,
    DB_PATH,
)


# ---------------------------------------------------------------------------
# flag_overpriced tests
# ---------------------------------------------------------------------------

class TestFlagOverpriced:
    def test_overpriced_above_threshold(self):
        record = {"price_pkr": 130.0, "drap_price_pkr": 100.0}
        result = flag_overpriced(record)
        assert result["overpriced"] == 1
        assert result["overprice_pct"] == 30.0

    def test_fair_below_threshold(self):
        record = {"price_pkr": 105.0, "drap_price_pkr": 100.0}
        result = flag_overpriced(record)
        assert result["overpriced"] == 0
        assert result["overprice_pct"] == 5.0

    def test_exact_threshold_not_overpriced(self):
        record = {"price_pkr": 110.0, "drap_price_pkr": 100.0}
        result = flag_overpriced(record)
        assert result["overpriced"] == 0
        assert result["overprice_pct"] == 10.0

    def test_price_below_drap(self):
        record = {"price_pkr": 80.0, "drap_price_pkr": 100.0}
        result = flag_overpriced(record)
        assert result["overpriced"] == 0
        assert result["overprice_pct"] == -20.0

    def test_zero_drap_price(self):
        record = {"price_pkr": 100.0, "drap_price_pkr": 0}
        result = flag_overpriced(record)
        assert result["overpriced"] == 0
        assert result["overprice_pct"] == 0.0

    def test_missing_drap_price(self):
        record = {"price_pkr": 100.0}
        result = flag_overpriced(record)
        assert result["overpriced"] == 0
        assert result["overprice_pct"] == 0.0


# ---------------------------------------------------------------------------
# load_drap_prices tests
# ---------------------------------------------------------------------------

class TestLoadDrapPrices:
    def test_returns_correct_count(self):
        medicines = load_drap_prices()
        assert len(medicines) >= 100

    def test_record_structure(self):
        medicines = load_drap_prices()
        for med in medicines:
            assert "name" in med
            assert "generic_name" in med
            assert "drap_price_pkr" in med
            assert isinstance(med["drap_price_pkr"], float)

    def test_prices_are_positive(self):
        medicines = load_drap_prices()
        for med in medicines:
            assert med["drap_price_pkr"] > 0


# ---------------------------------------------------------------------------
# generate_synthetic tests
# ---------------------------------------------------------------------------

class TestGenerateSynthetic:
    def test_record_count(self):
        drap = load_drap_prices()
        records = generate_synthetic(drap)
        # each medicine generates 6 records (2 current sources + 4 historical)
        assert len(records) == len(drap) * 6

    def test_record_fields(self):
        drap = load_drap_prices()
        records = generate_synthetic(drap)
        for rec in records:
            assert "name" in rec
            assert "price_pkr" in rec
            assert "drap_price_pkr" in rec
            assert "source" in rec
            assert "scraped_at" in rec
            assert "overpriced" in rec
            assert "availability" in rec

    def test_availability_values(self):
        drap = load_drap_prices()
        records = generate_synthetic(drap)
        valid = {"In Stock", "Limited", "Out of Stock"}
        for rec in records:
            assert rec["availability"] in valid, (
                f"Invalid availability: {rec['availability']}"
            )

    def test_prices_in_range(self):
        drap = load_drap_prices()
        records = generate_synthetic(drap)
        for rec in records:
            # Price should be between DRAP price and 1.5x DRAP (0-40% markup + some margin)
            assert rec["price_pkr"] >= rec["drap_price_pkr"] * 0.99
            assert rec["price_pkr"] <= rec["drap_price_pkr"] * 1.5


# ---------------------------------------------------------------------------
# init_db tests
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_creates_db_and_table(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test.db")
        monkeypatch.setattr("scraper.scrape.DB_PATH", db_path)

        init_db()

        assert os.path.exists(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(prices)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        assert "name" in columns
        assert "price_pkr" in columns
        assert "drap_price_pkr" in columns
        assert "overpriced" in columns
        assert "source" in columns
        assert "availability" in columns
        assert "scraped_at" in columns


# ---------------------------------------------------------------------------
# detect_anomalies tests
# ---------------------------------------------------------------------------

class TestDetectAnomalies:
    def test_with_few_records(self):
        records = [{"price_pkr": 100, "overprice_pct": 5}]
        result = detect_anomalies(records)
        assert result[0]["anomaly"] == 0

    def test_with_enough_records(self):
        records = [
            {"price_pkr": 100, "overprice_pct": 5},
            {"price_pkr": 105, "overprice_pct": 5},
            {"price_pkr": 102, "overprice_pct": 5},
            {"price_pkr": 98, "overprice_pct": 5},
            {"price_pkr": 103, "overprice_pct": 5},
            {"price_pkr": 500, "overprice_pct": 150},  # obvious outlier
        ]
        result = detect_anomalies(records)
        assert all("anomaly" in r for r in result)

    def test_adds_anomaly_key(self):
        records = [{"price_pkr": float(i), "overprice_pct": 0} for i in range(10)]
        result = detect_anomalies(records)
        for rec in result:
            assert "anomaly" in rec
            assert rec["anomaly"] in (0, 1)


# ---------------------------------------------------------------------------
# Pharmacy data tests
# ---------------------------------------------------------------------------

class TestPharmacyData:
    def _load_csv(self):
        rows = []
        with open(PHARMACIES_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows, reader.fieldnames

    def test_csv_exists(self):
        assert os.path.exists(PHARMACIES_CSV), "data/pharmacies.csv does not exist"

    def test_csv_structure(self):
        _, fieldnames = self._load_csv()
        required = {"name", "address", "city", "phone", "latitude", "longitude"}
        assert required.issubset(set(fieldnames)), (
            f"Missing columns: {required - set(fieldnames)}"
        )

    def test_csv_has_enough_rows(self):
        rows, _ = self._load_csv()
        assert len(rows) >= 20, f"Expected at least 20 pharmacies, got {len(rows)}"

    def test_cities_covered(self):
        rows, _ = self._load_csv()
        cities = {row["city"] for row in rows}
        expected = {"Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Faisalabad"}
        assert expected.issubset(cities), f"Missing cities: {expected - cities}"

    def test_coordinates_in_pakistan(self):
        rows, _ = self._load_csv()
        for row in rows:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            assert 24 <= lat <= 37, f"Latitude {lat} out of Pakistan bounds for {row['name']}"
            assert 61 <= lon <= 77, f"Longitude {lon} out of Pakistan bounds for {row['name']}"
