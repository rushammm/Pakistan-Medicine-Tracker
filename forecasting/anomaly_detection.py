"""
Per-Medicine Anomaly Detection
==============================
Isolation Forest trained per medicine on real scraped price history.
Generates plain English explanations for each detected anomaly.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")


@st.cache_data(ttl=300)
def detect_anomalies(medicine_name: str) -> pd.DataFrame | None:
    """Run Isolation Forest anomaly detection on a single medicine's price history.

    Features used:
    - price_pkr: the actual price
    - price_pct_change: % change from previous scrape (captures spikes)
    - days_since_start: temporal position (captures drift)

    Returns dataframe with columns: scraped_at, price_pkr, source,
    is_anomaly (bool), anomaly_score (float), or None if insufficient data.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT scraped_at, price_pkr, source, drap_price_pkr FROM prices WHERE name = ? ORDER BY scraped_at",
        conn, params=(medicine_name,),
    )
    conn.close()

    if len(df) < 5:
        return None

    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df["price_pct_change"] = df["price_pkr"].pct_change().fillna(0) * 100
    df["days_since_start"] = (df["scraped_at"] - df["scraped_at"].min()).dt.total_seconds() / 86400

    features = df[["price_pkr", "price_pct_change", "days_since_start"]].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # contamination=0.1 means we expect ~10% of observations to be anomalous
    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    model.fit(X)

    df["anomaly_score"] = -model.decision_function(X)  # higher = more anomalous
    df["is_anomaly"] = (model.predict(X) == -1)

    return df


def explain_anomaly(row: pd.Series, drap_price: float, med_avg_price: float) -> str:
    """Generate plain English explanation for a detected anomaly.

    Uses price context (DRAP reference, historical average) to explain
    why this data point was flagged.
    """
    price = row["price_pkr"]
    pct_change = row.get("price_pct_change", 0)
    date_str = pd.to_datetime(row["scraped_at"]).strftime("%b %d, %Y")

    parts = []

    # Price spike/drop
    if abs(pct_change) > 5:
        direction = "spiked" if pct_change > 0 else "dropped"
        parts.append(f"Price {direction} {abs(pct_change):.0f}% from previous scrape")

    # DRAP comparison
    if drap_price > 0:
        drap_dev = ((price - drap_price) / drap_price) * 100
        if drap_dev > 15:
            parts.append(f"{drap_dev:.0f}% above DRAP regulated price (Rs {drap_price:,.0f})")
        elif drap_dev < -20:
            parts.append(f"{abs(drap_dev):.0f}% below DRAP price — suspiciously cheap")

    # Average comparison
    if med_avg_price > 0:
        avg_dev = ((price - med_avg_price) / med_avg_price) * 100
        if abs(avg_dev) > 20:
            direction = "above" if avg_dev > 0 else "below"
            parts.append(f"{abs(avg_dev):.0f}% {direction} historical average (Rs {med_avg_price:,.0f})")

    if not parts:
        parts.append("Unusual pricing pattern detected by Isolation Forest")

    return f"{date_str}: " + ". ".join(parts) + "."


@st.cache_data(ttl=300)
def get_anomaly_summary() -> pd.DataFrame:
    """Run anomaly detection across ALL medicines in the database.

    Returns summary dataframe with: medicine, total_records, anomaly_count,
    anomaly_rate, worst_spike_pct, avg_price, drap_price.
    """
    conn = sqlite3.connect(DB_PATH)
    medicines = pd.read_sql_query("SELECT DISTINCT name FROM prices", conn)
    conn.close()

    rows = []
    for med_name in medicines["name"]:
        result = detect_anomalies(med_name)
        if result is None:
            continue

        anomaly_count = int(result["is_anomaly"].sum())
        total = len(result)

        rows.append({
            "Medicine": med_name,
            "Records": total,
            "Anomalies": anomaly_count,
            "Anomaly Rate": f"{anomaly_count / total:.0%}" if total > 0 else "0%",
            "Worst Spike": f"{result['price_pct_change'].abs().max():.1f}%",
            "Avg Price": f"Rs {result['price_pkr'].mean():,.0f}",
            "DRAP Price": f"Rs {result['drap_price_pkr'].iloc[0]:,.0f}" if result["drap_price_pkr"].iloc[0] > 0 else "N/A",
            "_anomaly_count": anomaly_count,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("_anomaly_count", ascending=False)
