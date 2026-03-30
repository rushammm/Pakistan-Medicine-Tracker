"""
ML Models Module
================
Pure ML backend — trains and serves models for medicine price intelligence.
No rendering code; UI is handled by supply_forecaster.py and app.py.

Models:
1. Stock-Out Risk Classifier (Random Forest)
2. Price Trend Predictor (Gradient Boosting Regressor)
3. Overpricing Anomaly Detector (Isolation Forest)
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc,
)
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Stock-Out Risk Classifier (Random Forest)
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=300)
def train_stockout_model(_df):
    """Train a Random Forest to predict stock-out risk.

    Features per medicine per time step:
    - price_pkr, drap_price_pkr, overprice_pct
    - source_encoded, availability_lag (previous availability)
    - price_change_pct (vs previous scrape)
    - generic_avg_price, brand_count

    Target: will this medicine be Out of Stock in the next scrape?
    """
    df = _df.copy()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df = df.sort_values(["name", "source", "scraped_at"])

    avail_map = {"In Stock": 0, "Limited": 1, "Out of Stock": 2, "Unknown": 1}
    df["avail_num"] = df["availability"].map(avail_map).fillna(1)

    df["next_avail"] = df.groupby(["name", "source"])["avail_num"].shift(-1)
    df["target"] = (df["next_avail"] >= 2).astype(int)

    labeled = df.dropna(subset=["next_avail"]).copy()
    if len(labeled) < 50:
        return None, None, None

    le_source = LabelEncoder()
    labeled["source_enc"] = le_source.fit_transform(labeled["source"])

    labeled["prev_price"] = labeled.groupby(["name", "source"])["price_pkr"].shift(1)
    labeled["price_change_pct"] = ((labeled["price_pkr"] - labeled["prev_price"]) / labeled["prev_price"] * 100).fillna(0)

    generic_stats = labeled.groupby("generic_name").agg(
        generic_avg_price=("price_pkr", "mean"),
        brand_count=("name", "nunique"),
    )
    labeled = labeled.merge(generic_stats, on="generic_name", how="left")

    feature_cols = [
        "price_pkr", "drap_price_pkr", "overprice_pct", "avail_num",
        "source_enc", "price_change_pct", "generic_avg_price", "brand_count",
    ]

    X = labeled[feature_cols].fillna(0)
    y = labeled["target"]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    )
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=min(5, len(y) // 10 + 1), scoring="f1")

    metrics = {
        "accuracy": model.score(X, y),
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "train_size": len(X),
        "positive_rate": y.mean(),
        "feature_importances": dict(zip(feature_cols, model.feature_importances_)),
    }

    return model, le_source, metrics


def predict_stockout(model, le_source, df, medicine_name):
    """Predict stock-out risk for a specific medicine using the trained model."""
    if model is None:
        return {"risk_score": 0.5, "risk_level": "Unknown", "model_type": "none"}

    med_df = df[df["name"] == medicine_name].sort_values("scraped_at")
    if med_df.empty:
        return {"risk_score": 0.5, "risk_level": "Unknown", "model_type": "none"}

    latest = med_df.iloc[-1]

    avail_map = {"In Stock": 0, "Limited": 1, "Out of Stock": 2, "Unknown": 1}

    prev = med_df.iloc[-2] if len(med_df) > 1 else latest
    price_change = ((latest["price_pkr"] - prev["price_pkr"]) / prev["price_pkr"] * 100) if prev["price_pkr"] > 0 else 0

    generic_meds = df[df["generic_name"] == latest.get("generic_name", "")]
    generic_avg = generic_meds["price_pkr"].mean() if not generic_meds.empty else latest["price_pkr"]
    brand_count = generic_meds["name"].nunique() if not generic_meds.empty else 1

    try:
        source_enc = le_source.transform([latest["source"]])[0]
    except (ValueError, KeyError):
        source_enc = 0

    features = np.array([[
        latest["price_pkr"],
        latest.get("drap_price_pkr", 0),
        latest.get("overprice_pct", 0),
        avail_map.get(latest.get("availability", "Unknown"), 1),
        source_enc,
        price_change,
        generic_avg,
        brand_count,
    ]])

    prob = model.predict_proba(features)[0]
    risk_score = prob[1] if len(prob) > 1 else 0

    if risk_score >= 0.6:
        risk_level = "High"
    elif risk_score >= 0.3:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "risk_score": round(float(risk_score), 3),
        "risk_level": risk_level,
        "model_type": "Random Forest",
    }


# ---------------------------------------------------------------------------
# 2. Price Trend Predictor (Gradient Boosting)
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=300)
def train_price_model(_df):
    """Train a Gradient Boosting model to predict next price."""
    df = _df.copy()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df = df.sort_values(["name", "source", "scraped_at"])

    df["next_price"] = df.groupby(["name", "source"])["price_pkr"].shift(-1)
    labeled = df.dropna(subset=["next_price"]).copy()

    if len(labeled) < 50:
        return None, None, None

    le = LabelEncoder()
    labeled["name_enc"] = le.fit_transform(labeled["name"])

    avail_map = {"In Stock": 0, "Limited": 1, "Out of Stock": 2, "Unknown": 1}
    labeled["avail_num"] = labeled["availability"].map(avail_map).fillna(1)

    labeled["prev_price"] = labeled.groupby(["name", "source"])["price_pkr"].shift(1)
    labeled["price_change"] = (labeled["price_pkr"] - labeled["prev_price"].fillna(labeled["price_pkr"]))

    feature_cols = ["price_pkr", "drap_price_pkr", "overprice_pct", "avail_num", "name_enc", "price_change"]
    X = labeled[feature_cols].fillna(0)
    y = labeled["next_price"]

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=min(5, len(y) // 10 + 1), scoring="r2")

    metrics = {
        "r2": model.score(X, y),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "train_size": len(X),
        "feature_importances": dict(zip(feature_cols, model.feature_importances_)),
    }

    return model, le, metrics


def predict_prices(model, le, df):
    """Predict next-period prices for all medicines using the trained model."""
    if model is None:
        return pd.DataFrame()

    df = df.copy()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last").copy()

    if latest.empty:
        return pd.DataFrame()

    avail_map = {"In Stock": 0, "Limited": 1, "Out of Stock": 2, "Unknown": 1}
    latest["avail_num"] = latest["availability"].map(avail_map).fillna(1)

    known_classes = set(le.classes_)
    latest["name_enc"] = latest["name"].apply(lambda n: le.transform([n])[0] if n in known_classes else -1)

    prev = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")
    second = df.sort_values("scraped_at").groupby(["name", "source"]).nth(-2)
    if not second.empty:
        prev_prices = second.reset_index()[["name", "source", "price_pkr"]].rename(columns={"price_pkr": "prev_price"})
        latest = latest.merge(prev_prices, on=["name", "source"], how="left")
        latest["price_change"] = (latest["price_pkr"] - latest["prev_price"].fillna(latest["price_pkr"]))
    else:
        latest["price_change"] = 0

    feature_cols = ["price_pkr", "drap_price_pkr", "overprice_pct", "avail_num", "name_enc", "price_change"]
    X = latest[feature_cols].fillna(0)

    latest["predicted_price"] = model.predict(X)
    latest["price_diff"] = latest["predicted_price"] - latest["price_pkr"]
    latest["price_diff_pct"] = ((latest["price_diff"] / latest["price_pkr"]) * 100)

    result = latest[["name", "brand", "generic_name", "price_pkr", "drap_price_pkr",
                      "predicted_price", "price_diff", "price_diff_pct", "source"]].copy()
    result.columns = ["Medicine", "Brand", "Generic", "Current Price", "DRAP Price",
                       "Predicted Price", "Change (PKR)", "Change (%)", "Source"]
    return result.sort_values("Change (%)", ascending=False)


# ---------------------------------------------------------------------------
# 3. Overpricing Anomaly Detector (Isolation Forest)
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=300)
def train_anomaly_model(_df):
    """Train Isolation Forest for price anomaly detection."""
    df = _df.copy()
    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")

    if len(latest) < 10:
        return None, None, None

    features_df = latest[["price_pkr", "overprice_pct", "drap_price_pkr"]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=200)
    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    latest = latest.copy()
    latest["anomaly_score"] = -scores
    latest["is_anomaly"] = (predictions == -1).astype(int)

    metrics = {
        "total": len(latest),
        "anomalies": int(latest["is_anomaly"].sum()),
        "contamination": 0.1,
    }

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# 4. Stock-Out Model Evaluation (for Supply Forecaster transparency)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def evaluate_stockout_model(_df):
    """Get detailed evaluation metrics with train/test split."""
    df = _df.copy()
    df["scraped_at"] = pd.to_datetime(df["scraped_at"])
    df = df.sort_values(["name", "source", "scraped_at"])

    avail_map = {"In Stock": 0, "Limited": 1, "Out of Stock": 2, "Unknown": 1}
    df["avail_num"] = df["availability"].map(avail_map).fillna(1)
    df["next_avail"] = df.groupby(["name", "source"])["avail_num"].shift(-1)
    df["target"] = (df["next_avail"] >= 2).astype(int)

    labeled = df.dropna(subset=["next_avail"]).copy()
    if len(labeled) < 50:
        return None

    le_source = LabelEncoder()
    labeled["source_enc"] = le_source.fit_transform(labeled["source"])
    labeled["prev_price"] = labeled.groupby(["name", "source"])["price_pkr"].shift(1)
    labeled["price_change_pct"] = ((labeled["price_pkr"] - labeled["prev_price"]) / labeled["prev_price"] * 100).fillna(0)

    generic_stats = labeled.groupby("generic_name").agg(
        generic_avg_price=("price_pkr", "mean"), brand_count=("name", "nunique"),
    )
    labeled = labeled.merge(generic_stats, on="generic_name", how="left")

    feature_cols = ["price_pkr", "drap_price_pkr", "overprice_pct", "avail_num",
                    "source_enc", "price_change_pct", "generic_avg_price", "brand_count"]
    X = labeled[feature_cols].fillna(0)
    y = labeled["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc,
        "fpr": fpr, "tpr": tpr,
        "confusion_matrix": cm,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "feature_importances": dict(zip(feature_cols, model.feature_importances_)),
    }
