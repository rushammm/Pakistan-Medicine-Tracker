"""
Unit tests for the prescription scanner helpers.
Run with: python -m pytest tests/test_scanner.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import (
    clean_prescription_line,
    match_medicine,
    extract_medicines_from_text,
    load_drap_lookup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def drap_data():
    """Load real DRAP data once for all tests."""
    # load_drap_lookup is cached by streamlit, call the underlying logic
    import csv as _csv
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "drap_prices.csv")
    all_names = []
    generic_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            generic = row.get("generic_name", "").strip()
            all_names.append(name)
            if generic:
                generic_map.setdefault(generic.lower(), []).append(name)
    return all_names, generic_map


# ---------------------------------------------------------------------------
# clean_prescription_line
# ---------------------------------------------------------------------------

class TestCleanPrescriptionLine:
    def test_strips_tab_prefix(self):
        assert clean_prescription_line("Tab. Panadol 500mg") == "Panadol 500mg"

    def test_strips_cap_prefix(self):
        assert clean_prescription_line("Cap. Amoxil 500mg") == "Amoxil 500mg"

    def test_strips_syp_prefix(self):
        assert clean_prescription_line("Syp. Calpol Pediatric Syrup") == "Calpol Pediatric Syrup"

    def test_strips_inj_prefix(self):
        assert clean_prescription_line("Inj. Augmentin") == "Augmentin"

    def test_strips_dosage_suffix_1x3(self):
        result = clean_prescription_line("Panadol 500mg 1x3")
        assert "1x3" not in result
        assert "Panadol 500mg" in result

    def test_strips_bd_suffix(self):
        result = clean_prescription_line("Amoxil 500mg BD")
        assert "BD" not in result
        assert "Amoxil 500mg" in result

    def test_strips_tds_suffix(self):
        result = clean_prescription_line("Brufen 400mg TDS")
        assert "TDS" not in result

    def test_strips_for_days(self):
        result = clean_prescription_line("Calpol Syrup for 5 days")
        assert "for 5 days" not in result

    def test_strips_prefix_and_suffix(self):
        result = clean_prescription_line("Tab. Panadol 500mg 1x3")
        assert "Tab." not in result
        assert "1x3" not in result
        assert "Panadol 500mg" in result

    def test_empty_string(self):
        assert clean_prescription_line("") == ""

    def test_plain_name_unchanged(self):
        assert clean_prescription_line("Panadol 500mg") == "Panadol 500mg"


# ---------------------------------------------------------------------------
# match_medicine
# ---------------------------------------------------------------------------

class TestMatchMedicine:
    def test_exact_match(self, drap_data):
        all_names, generic_map = drap_data
        results = match_medicine("Panadol 500mg", all_names, generic_map)
        assert len(results) >= 1
        assert results[0]["name"] == "Panadol 500mg"
        assert results[0]["match_type"] == "exact"
        assert results[0]["score"] == 1.0

    def test_exact_match_case_insensitive(self, drap_data):
        all_names, generic_map = drap_data
        results = match_medicine("panadol 500mg", all_names, generic_map)
        assert len(results) >= 1
        assert results[0]["name"] == "Panadol 500mg"
        assert results[0]["score"] == 1.0

    def test_fuzzy_match(self, drap_data):
        all_names, generic_map = drap_data
        # Misspelling of Panadol
        results = match_medicine("Panadl 500mg", all_names, generic_map)
        assert len(results) >= 1
        assert results[0]["match_type"] in ("fuzzy", "exact")
        assert "Panadol" in results[0]["name"]

    def test_generic_match(self, drap_data):
        all_names, generic_map = drap_data
        results = match_medicine("Paracetamol", all_names, generic_map)
        assert len(results) >= 1
        # Should find brands containing Paracetamol
        names = [r["name"] for r in results]
        assert any("Panadol" in n or "Paracetamol" in n for n in names)

    def test_empty_text(self, drap_data):
        all_names, generic_map = drap_data
        results = match_medicine("", all_names, generic_map)
        assert results == []

    def test_no_match_high_confidence(self, drap_data):
        all_names, generic_map = drap_data
        results = match_medicine("xyznonexistent123", all_names, generic_map)
        # May return low-score fuzzy matches; none should be high confidence
        high_conf = [r for r in results if r["score"] >= 0.6]
        assert high_conf == []

    def test_ocr_misspelling_brufen(self, drap_data):
        """OCR might read 'Brufen' as 'Brulen' — should still fuzzy match."""
        all_names, generic_map = drap_data
        results = match_medicine("Brulen 400mg", all_names, generic_map)
        names = [r["name"] for r in results]
        assert any("Brufen" in n for n in names)

    def test_ocr_partial_word_match(self, drap_data):
        """OCR might read 'Calpol' as 'Calpo' — first-word matching should help."""
        all_names, generic_map = drap_data
        results = match_medicine("Calpo", all_names, generic_map)
        names = [r["name"] for r in results]
        assert any("Calpol" in n for n in names)


# ---------------------------------------------------------------------------
# extract_medicines_from_text
# ---------------------------------------------------------------------------

class TestExtractMedicinesFromText:
    def test_single_line(self, drap_data):
        all_names, generic_map = drap_data
        results = extract_medicines_from_text("Panadol 500mg", all_names, generic_map)
        assert len(results) == 1
        assert results[0]["matches"][0]["name"] == "Panadol 500mg"

    def test_multiple_lines(self, drap_data):
        all_names, generic_map = drap_data
        text = "Tab. Panadol 500mg 1x3\nCap. Amoxil 500mg BD"
        results = extract_medicines_from_text(text, all_names, generic_map)
        assert len(results) >= 1
        matched_names = [r["matches"][0]["name"] for r in results]
        assert "Panadol 500mg" in matched_names

    def test_with_rx_prefixes(self, drap_data):
        all_names, generic_map = drap_data
        text = "Tab. Brufen 400mg TDS"
        results = extract_medicines_from_text(text, all_names, generic_map)
        assert len(results) >= 1
        assert "Brufen 400mg" in results[0]["matches"][0]["name"]

    def test_deduplication(self, drap_data):
        all_names, generic_map = drap_data
        text = "Panadol 500mg\nPanadol 500mg"
        results = extract_medicines_from_text(text, all_names, generic_map)
        all_matched = []
        for r in results:
            for m in r["matches"]:
                all_matched.append(m["name"])
        # Panadol 500mg should appear only once across all results
        assert all_matched.count("Panadol 500mg") == 1

    def test_empty_text(self, drap_data):
        all_names, generic_map = drap_data
        results = extract_medicines_from_text("", all_names, generic_map)
        assert results == []

    def test_blank_lines_ignored(self, drap_data):
        all_names, generic_map = drap_data
        text = "\n\n  \nPanadol 500mg\n\n"
        results = extract_medicines_from_text(text, all_names, generic_map)
        assert len(results) == 1

    def test_result_structure(self, drap_data):
        all_names, generic_map = drap_data
        results = extract_medicines_from_text("Panadol 500mg", all_names, generic_map)
        entry = results[0]
        assert "line" in entry
        assert "cleaned" in entry
        assert "matches" in entry
        match = entry["matches"][0]
        assert "name" in match
        assert "match_type" in match
        assert "score" in match
