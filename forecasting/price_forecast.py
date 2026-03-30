"""
Price Forecast Module
=====================
Prophet-based price forecasting trained on real scraped price history.
Never interpolates or fakes data — shows honest "not enough data" messages
when history is insufficient.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Minimum data points needed for Prophet to produce meaningful forecasts.
# Prophet technically works with fewer, but results are unreliable below this.
MIN_DATA_POINTS = 14


@st.cache_data(ttl=300)
def load_price_history(medicine_name: str) -> pd.DataFrame | None:
    """Load real scraped price history from SQLite for a given medicine.

    Returns dataframe with columns: ds (datetime), y (float price_pkr),
    aggregated to one price per day (mean across sources).
    Returns None if fewer than MIN_DATA_POINTS available.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DATE(scraped_at) as ds, AVG(price_pkr) as y
        FROM prices
        WHERE name = ?
        GROUP BY DATE(scraped_at)
        ORDER BY ds
    """
    df = pd.read_sql_query(query, conn, params=(medicine_name,))
    conn.close()

    if df.empty or len(df) < MIN_DATA_POINTS:
        return None

    df["ds"] = pd.to_datetime(df["ds"])
    return df


def train_prophet_model(df: pd.DataFrame, medicine_name: str):
    """Train Prophet on real price history.

    Uses yearly + weekly seasonality. Daily seasonality is off because
    we aggregate to daily prices. Pakistani public holidays added as
    special events since pharmacy demand shifts around Eid, etc.

    Returns trained Prophet model, or None if Prophet is not installed.
    """
    try:
        from prophet import Prophet
    except ImportError:
        return None

    # Pakistani public holidays (approximate — dates shift yearly for Islamic holidays)
    holidays = pd.DataFrame({
        "holiday": [
            "Pakistan Day", "Labour Day", "Independence Day", "Iqbal Day",
            "Quaid-e-Azam Day", "Eid ul-Fitr (approx)", "Eid ul-Adha (approx)",
        ],
        "ds": pd.to_datetime([
            "2026-03-23", "2026-05-01", "2026-08-14", "2026-11-09",
            "2026-12-25", "2026-03-31", "2026-06-07",
        ]),
        "lower_window": 0,
        "upper_window": 1,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # data is aggregated daily, not sub-daily
        holidays=holidays,
        interval_width=0.9,  # 90% confidence interval
        changepoint_prior_scale=0.05,  # conservative — avoid overfitting on few points
    )
    model.fit(df)

    # Save model
    try:
        import joblib
        safe_name = medicine_name.replace(" ", "_").replace("/", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"prophet_{safe_name}.pkl"))
    except Exception:
        pass

    return model


@st.cache_data(ttl=300)
def forecast_price(medicine_name: str, days: int = 7) -> tuple[pd.DataFrame | None, pd.DataFrame | None, int]:
    """Forecast price for the next N days using Prophet.

    Returns:
        (history_df, forecast_df, data_point_count)
        history_df: real scraped data with columns ds, y
        forecast_df: Prophet forecast with ds, yhat, yhat_lower, yhat_upper, trend
        data_point_count: number of real data points used
    """
    history = load_price_history(medicine_name)
    if history is None:
        # Return count of available points for the "not enough data" message
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute(
            "SELECT COUNT(DISTINCT DATE(scraped_at)) FROM prices WHERE name = ?",
            (medicine_name,),
        ).fetchone()[0]
        conn.close()
        return None, None, count

    model = train_prophet_model(history, medicine_name)
    if model is None:
        return history, None, len(history)

    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)

    forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    # Clamp to non-negative — prices can't go below zero
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast_out[col] = forecast_out[col].clip(lower=0)
    return history, forecast_out, len(history)


def get_forecast_insight(forecast_df: pd.DataFrame, current_price: float) -> dict:
    """Generate plain English insight from forecast.

    Compares the last forecasted price against the current price to
    determine direction and magnitude.
    """
    if forecast_df is None or forecast_df.empty or current_price <= 0:
        return {
            "direction": "unknown",
            "pct_change": 0,
            "confidence": 0,
            "message": "Insufficient data for forecast insight.",
        }

    last_forecast = forecast_df.iloc[-1]
    predicted = last_forecast["yhat"]
    lower = last_forecast["yhat_lower"]
    upper = last_forecast["yhat_upper"]

    pct_change = ((predicted - current_price) / current_price) * 100
    confidence_width = ((upper - lower) / current_price) * 100

    if pct_change > 2:
        direction = "rising"
        message = f"Price predicted to rise ~{pct_change:.1f}% over the next 7 days."
    elif pct_change < -2:
        direction = "falling"
        message = f"Price may drop ~{abs(pct_change):.1f}% over the next 7 days."
    else:
        direction = "stable"
        message = "Price expected to remain stable over the next 7 days."

    return {
        "direction": direction,
        "pct_change": round(pct_change, 1),
        "confidence": round(100 - confidence_width, 1),
        "message": message,
    }
