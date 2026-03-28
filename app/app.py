"""
Pakistan Medicine Price Tracker — Streamlit Dashboard
=====================================================
Run with:  streamlit run app/app.py
"""

import os
import sys
import sqlite3
from datetime import datetime
from urllib.parse import quote
import re
import csv
from difflib import get_close_matches, SequenceMatcher

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")
PHARMACIES_CSV = os.path.join(BASE_DIR, "data", "pharmacies.csv")
sys.path.insert(0, BASE_DIR)

DAWAAI_SEARCH_URL = "https://dawaai.pk/search?q={query}"
MEDSTORE_SEARCH_URL = "https://medstore.com.pk/catalogsearch/result/?q={query}"

# ---------------------------------------------------------------------------
# Color palette — monochrome (zinc scale)
# ---------------------------------------------------------------------------

COLORS = {
    "bg":         "#0a0a0f",       # deep dark
    "surface":    "rgba(255,255,255,0.05)",  # glass
    "surface2":   "rgba(255,255,255,0.08)",  # glass elevated
    "border":     "rgba(255,255,255,0.08)",  # subtle border
    "border_l":   "rgba(255,255,255,0.15)",  # hover border
    "text":       "#e4e4e7",       # zinc-200
    "text_dim":   "#71717a",       # zinc-500
    "accent":     "#fafafa",       # zinc-50
    "accent_dim": "#52525b",       # zinc-600
    "blue":       "#e4e4e7",       # zinc-200
    "blue_dim":   "rgba(255,255,255,0.06)",
    "blue_light": "#a1a1aa",       # zinc-400
    "green":      "#fafafa",       # zinc-50
    "green_dim":  "rgba(255,255,255,0.04)",
    "green_light":"#a1a1aa",       # zinc-400
}

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedTracker PK",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {{
        background-color: {COLORS["bg"]};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* ---- Glass mixin ---- */
    /* Shared glass properties used across cards, hero, sidebar, etc. */

    /* Header */
    .hero {{
        background: {COLORS["surface"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
    }}
    .hero h1 {{
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.03em;
    }}
    .hero p {{
        color: {COLORS["text_dim"]};
        font-size: 0.9rem;
        margin: 0;
        font-weight: 400;
    }}

    /* Metric cards */
    .metric-row {{
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        flex: 1;
        background: {COLORS["surface"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: border-color 0.2s, background 0.2s;
    }}
    .metric-card:hover {{
        border-color: {COLORS["border_l"]};
        background: {COLORS["surface2"]};
    }}
    .metric-label {{
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLORS["text_dim"]};
        margin-bottom: 0.5rem;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }}
    .metric-value.blue {{ color: {COLORS["blue"]}; }}
    .metric-value.lblue {{ color: {COLORS["blue_light"]}; }}
    .metric-value.green {{ color: {COLORS["green"]}; }}
    .metric-sub {{
        font-size: 0.75rem;
        color: {COLORS["text_dim"]};
        margin-top: 0.35rem;
    }}

    /* Section headers */
    .section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.01em;
    }}

    /* Status pills */
    .pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .pill-blue {{
        background: {COLORS["blue_dim"]};
        color: {COLORS["blue"]};
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .pill-lblue {{
        background: {COLORS["blue_dim"]};
        color: {COLORS["blue_light"]};
        border: 1px solid rgba(255,255,255,0.08);
    }}
    .pill-green {{
        background: {COLORS["green_dim"]};
        color: {COLORS["green"]};
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .pill-gray {{
        background: rgba(255,255,255,0.04);
        color: {COLORS["text_dim"]};
        border: 1px solid rgba(255,255,255,0.06);
    }}

    /* Divider */
    .divider {{
        border: none;
        border-top: 1px solid {COLORS["border"]};
        margin: 2rem 0;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 2rem 0 1rem;
        color: {COLORS["text_dim"]};
        font-size: 0.8rem;
    }}
    .footer a {{
        color: {COLORS["accent"]};
        text-decoration: none;
    }}

    /* Hide Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: #1a1a1f !important;
        border-right: 1px solid {COLORS["border"]};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background: #1a1a1f !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS["text"]} !important;
    }}

    /* Plotly chart backgrounds */
    .stPlotlyChart {{
        background: {COLORS["surface"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 0.5rem;
    }}

    /* Dataframe */
    .stDataFrame {{
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* Streamlit metric override — hide default */
    div[data-testid="stMetric"] {{
        display: none;
    }}

    /* Input fields */
    .stTextInput > div > div > input {{
        background: {COLORS["surface2"]} !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text"]} !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: rgba(255,255,255,0.25) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.1) !important;
    }}

    /* Multiselect */
    .stMultiSelect > div > div {{
        background: {COLORS["surface2"]} !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
    }}

    /* Download buttons */
    .stDownloadButton > button {{
        background: {COLORS["surface"]} !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}
    .stDownloadButton > button:hover {{
        border-color: {COLORS["border_l"]} !important;
        background: {COLORS["surface2"]} !important;
        color: #ffffff !important;
    }}

    /* Cheapest deal card */
    .deal-card {{
        background: {COLORS["surface"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0 1.5rem;
    }}
    .deal-card .deal-title {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {COLORS["text_dim"]};
        margin-bottom: 0.6rem;
    }}
    .deal-card .deal-medicine {{
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
    }}
    .deal-card .deal-detail {{
        color: {COLORS["text"]};
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }}
    .deal-card .deal-saving {{
        color: {COLORS["accent"]};
        font-weight: 600;
    }}

    /* WhatsApp share link */
    .wa-btn {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        background: {COLORS["surface"]};
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text"]} !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.2s;
        margin-top: 0.5rem;
    }}
    .wa-btn:hover {{
        border-color: {COLORS["border_l"]};
        background: {COLORS["surface2"]};
    }}

    /* Cart summary card */
    .cart-summary {{
        background: {COLORS["surface"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.5rem;
    }}
    .cart-total {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }}
    .cart-winner {{
        border-color: rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.07);
    }}

    /* Dialog / modal */
    div[data-testid="stModal"] > div {{
        background: rgba(15,15,20,0.85) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid {COLORS["border_l"]} !important;
        border-radius: 16px !important;
        color: {COLORS["text"]} !important;
    }}
    div[data-testid="stModal"] p,
    div[data-testid="stModal"] label,
    div[data-testid="stModal"] span {{
        color: {COLORS["text"]} !important;
    }}
    div[data-testid="stModal"] strong {{
        color: #ffffff !important;
    }}
    div[data-testid="stModal"] button[kind="primary"] {{
        background: {COLORS["surface2"]} !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid {COLORS["border_l"]} !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: border-color 0.2s !important;
    }}
    div[data-testid="stModal"] button[kind="primary"]:hover {{
        border-color: {COLORS["text_dim"]} !important;
    }}
    div[data-testid="stModal"] [data-testid="stSelectbox"] > div > div {{
        background: {COLORS["surface2"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text"]} !important;
    }}
    /* Dialog backdrop */
    div[data-testid="stModal"]::backdrop {{
        background: rgba(0, 0, 0, 0.6) !important;
        backdrop-filter: blur(4px) !important;
    }}
    /* Selectbox dropdown list */
    div[data-testid="stModal"] [data-baseweb="popover"] {{
        background: rgba(15,15,20,0.9) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid {COLORS["border_l"]} !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stModal"] [data-baseweb="popover"] li {{
        color: {COLORS["text"]} !important;
    }}
    div[data-testid="stModal"] [data-baseweb="popover"] li:hover {{
        background: {COLORS["surface2"]} !important;
    }}

    /* Scanner card */
    .scanner-card {{
        background: {COLORS["surface"]};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 0.75rem;
    }}
    .scanner-match {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    .scanner-match:last-child {{
        border-bottom: none;
    }}
    .scanner-match-name {{
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
    }}
    .scanner-match-detail {{
        color: {COLORS["text_dim"]};
        font-size: 0.8rem;
    }}

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: {COLORS["surface"]};
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS["border"]};
        border-radius: 10px;
        padding: 4px;
        gap: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {COLORS["text_dim"]} !important;
        font-weight: 500;
        padding: 6px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS["surface2"]} !important;
        color: #ffffff !important;
    }}

    /* Streamlit buttons */
    .stButton > button {{
        background: {COLORS["surface"]} !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}
    .stButton > button:hover {{
        border-color: {COLORS["border_l"]} !important;
        background: {COLORS["surface2"]} !important;
        color: #ffffff !important;
    }}

    /* Streamlit expander */
    .streamlit-expanderHeader {{
        background: {COLORS["surface"]} !important;
        backdrop-filter: blur(12px);
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 10px !important;
        color: {COLORS["text"]} !important;
    }}

    /* Text area */
    .stTextArea textarea {{
        background: {COLORS["surface2"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text"]} !important;
    }}

    /* Selectbox */
    .stSelectbox > div > div {{
        background: {COLORS["surface2"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
    }}
    [data-baseweb="popover"] {{
        background: rgba(15,15,20,0.9) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid {COLORS["border_l"]} !important;
        border-radius: 8px !important;
    }}

    /* Reduce default Streamlit block spacing */
    .stElementContainer {{
        margin-bottom: 0 !important;
    }}
    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.5rem !important;
    }}

    /* Alerts / info boxes */
    .stAlert {{
        background: {COLORS["surface"]} !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 10px !important;
    }}

    /* Multiselect tags — grey glass */
    .stMultiSelect [data-baseweb="tag"] {{
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 6px !important;
        color: #e4e4e7 !important;
    }}
    .stMultiSelect [data-baseweb="tag"] span {{
        color: #e4e4e7 !important;
    }}
    .stMultiSelect [data-baseweb="tag"] [role="presentation"] {{
        color: #a1a1aa !important;
    }}
</style>
""", unsafe_allow_html=True)

# Plotly template
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    title_font=dict(size=14, color="#ffffff"),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(
        bgcolor=COLORS["surface2"],
        bordercolor=COLORS["border"],
        font_color=COLORS["text"],
    ),
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM prices ORDER BY scraped_at DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def ensure_data():
    # Guard: don't re-scrape on every Streamlit rerun within the same session
    if st.session_state.get("_scrape_done"):
        return

    if not os.path.exists(DB_PATH):
        needs_scrape = True
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            conn.close()
            needs_scrape = count == 0
        except Exception:
            needs_scrape = True

    if needs_scrape:
        with st.spinner("Initializing data — this may take a moment on first run..."):
            from scraper.scrape import run_scraper
            run_scraper()
        st.cache_data.clear()

    st.session_state["_scrape_done"] = True


def detect_anomalies_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 5:
        df["anomaly"] = 0
        return df
    features = df[["price_pkr", "overprice_pct"]].fillna(0).values
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(features)
    df["anomaly"] = (preds == -1).astype(int)
    return df


@st.cache_data
def load_pharmacies() -> pd.DataFrame:
    if not os.path.exists(PHARMACIES_CSV):
        return pd.DataFrame()
    return pd.read_csv(PHARMACIES_CSV)


def whatsapp_link(text: str) -> str:
    """Return a WhatsApp share URL for the given text."""
    return f"https://wa.me/?text={quote(text)}"


# ---------------------------------------------------------------------------
# Prescription scanner helpers
# ---------------------------------------------------------------------------

DRAP_CSV = os.path.join(BASE_DIR, "data", "drap_prices.csv")

@st.cache_data
def load_drap_lookup():
    """Read drap_prices.csv and return all_names list + generic_map dict."""
    all_names = []
    generic_map: dict[str, list[str]] = {}
    if not os.path.exists(DRAP_CSV):
        return all_names, generic_map
    with open(DRAP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            generic = row.get("generic_name", "").strip()
            all_names.append(name)
            if generic:
                generic_map.setdefault(generic.lower(), []).append(name)
    return all_names, generic_map


_RX_PREFIX = re.compile(
    r"^\s*(?:Tab\.?|Cap\.?|Syp\.?|Inj\.?|Susp\.?|Cr\.?|Oint\.?)\s*",
    re.IGNORECASE,
)
_RX_SUFFIX = re.compile(
    r"\s+(?:\d+\s*x\s*\d+|BD|TDS|OD|QID|PRN|SOS|HS|for\s+\d+\s*days?|after\s+meal|before\s+meal|daily|weekly)\s*",
    re.IGNORECASE,
)

def clean_prescription_line(line: str) -> str:
    """Strip common Rx prefixes/suffixes from a prescription line."""
    line = _RX_PREFIX.sub("", line)
    line = _RX_SUFFIX.sub(" ", line)
    return line.strip()


def match_medicine(text: str, all_names: list[str], generic_map: dict[str, list[str]]) -> list[dict]:
    """Three-tier matching: exact → fuzzy → generic. Returns list of match dicts."""
    text_lower = text.lower().strip()
    if not text_lower:
        return []

    results = []

    # 1. Exact match (case-insensitive)
    for name in all_names:
        if name.lower() == text_lower:
            results.append({"name": name, "match_type": "exact", "score": 1.0})
            return results

    # 2. Fuzzy match on full name
    fuzzy = get_close_matches(text_lower, [n.lower() for n in all_names], n=3, cutoff=0.4)
    name_lower_map = {n.lower(): n for n in all_names}
    for match_lower in fuzzy:
        original = name_lower_map[match_lower]
        score = SequenceMatcher(None, text_lower, match_lower).ratio()
        results.append({"name": original, "match_type": "fuzzy", "score": round(score, 2)})

    # 3. Fuzzy match on first word of each medicine name (helps with OCR errors)
    if not results:
        first_words = {}
        for name in all_names:
            fw = name.split()[0].lower()
            first_words.setdefault(fw, []).append(name)
        text_first = text_lower.split()[0] if text_lower.split() else text_lower
        fw_matches = get_close_matches(text_first, list(first_words.keys()), n=3, cutoff=0.4)
        for fw in fw_matches:
            score = SequenceMatcher(None, text_first, fw).ratio()
            for name in first_words[fw]:
                if not any(r["name"] == name for r in results):
                    results.append({"name": name, "match_type": "fuzzy", "score": round(score * 0.9, 2)})

    # 4. Generic name match
    for generic_lower, brand_names in generic_map.items():
        if generic_lower in text_lower or text_lower in generic_lower:
            for bname in brand_names:
                if not any(r["name"] == bname for r in results):
                    results.append({"name": bname, "match_type": "generic", "score": 0.7})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def extract_medicines_from_text(
    raw_text: str, all_names: list[str], generic_map: dict[str, list[str]]
) -> list[dict]:
    """Split text into lines, clean each, match each, deduplicate."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    seen = set()
    results = []
    for line in lines:
        cleaned = clean_prescription_line(line)
        if not cleaned:
            continue
        matches = match_medicine(cleaned, all_names, generic_map)
        # Deduplicate across lines
        unique_matches = []
        for m in matches:
            if m["name"] not in seen:
                unique_matches.append(m)
                seen.add(m["name"])
        if unique_matches:
            results.append({"line": line, "cleaned": cleaned, "matches": unique_matches})
    return results


# ---------------------------------------------------------------------------
# Location dialog
# ---------------------------------------------------------------------------

PHARMACY_CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Faisalabad"]

@st.dialog("Where are you located?")
def location_dialog():
    st.markdown("Select your city so we can show you the **nearest pharmacies**.")
    city = st.selectbox("Your city", options=PHARMACY_CITIES, key="loc_dialog_city")
    if st.button("Confirm", type="primary", use_container_width=True):
        st.session_state["user_city"] = city
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("### MedTracker PK")

    if st.sidebar.button("Refresh Data", width="stretch"):
        with st.spinner("Scraping prices..."):
            from scraper.scrape import run_scraper
            run_scraper()
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("")

    pharmacies_df = load_pharmacies()
    cities = ["All"] + sorted(pharmacies_df["city"].unique().tolist()) if not pharmacies_df.empty else ["All"]
    user_city = st.session_state.get("user_city", "All")
    default_idx = cities.index(user_city) if user_city in cities else 0
    selected_city = st.sidebar.selectbox("Pharmacy City", cities, index=default_idx)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built by [Rusham Elahi](https://github.com/rushammm)")

    return selected_city


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_data()
    df = load_data()

    # --- Location popup (first visit only) ---
    if "user_city" not in st.session_state:
        location_dialog()

    # --- Hero (reworded) ---
    st.markdown("""
    <div class="hero">
        <h1>MedTracker PK</h1>
        <p>Find cheaper alternatives for your medicines</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available. Click **Refresh Data** in the sidebar.")
        render_sidebar(df)
        return

    df = detect_anomalies_df(df)
    if "availability" not in df.columns:
        df["availability"] = "Unknown"
    selected_city = render_sidebar(df)

    all_medicines = sorted(df["name"].unique().tolist())

    # =================================================================
    # TOP-LEVEL NAVIGATION TABS
    # =================================================================
    tab_lookup, tab_rx, tab_calc, tab_pharm, tab_analytics, tab_research = st.tabs([
        "Medicine Lookup",
        "Prescription Cart",
        "Cost Calculator",
        "Pharmacies",
        "Analytics",
        "Research Insights",
    ])

    # =================================================================
    # 1. MEDICINE LOOKUP
    # =================================================================
    with tab_lookup:
        st.caption("Search any medicine to find cheaper alternatives and compare prices across pharmacies.")

        selected_medicine = st.selectbox(
            "Select a medicine to find cheaper alternatives",
            options=all_medicines,
            key="medicine_lookup",
        )

        if selected_medicine:
            med_row = df[df["name"] == selected_medicine].sort_values("scraped_at").iloc[-1]
            selected_price = med_row["price_pkr"]
            generic = med_row.get("generic_name", "")

            if generic:
                st.markdown(
                    f'Active ingredient: <span class="pill pill-green">{generic}</span>',
                    unsafe_allow_html=True,
                )

                alternatives = df[
                    df["generic_name"].str.lower() == generic.lower()
                ].sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")

                cheapest = alternatives.sort_values("price_pkr").iloc[0]
                saving = selected_price - cheapest["price_pkr"]

                # a) "Best Deal" card
                if cheapest["name"] != selected_medicine and saving > 0:
                    wa_text = (
                        f"Switch from {selected_medicine} (Rs {selected_price:.0f}) to "
                        f"{cheapest['name']} (Rs {cheapest['price_pkr']:.0f}) — "
                        f"Save Rs {saving:.0f}! Same formula: {generic}. "
                        f"Tracked on MedTracker PK."
                    )
                    search_term = cheapest["name"]
                    dawaai_url = DAWAAI_SEARCH_URL.format(query=search_term.replace(" ", "+"))
                    medstore_url = MEDSTORE_SEARCH_URL.format(query=search_term.replace(" ", "+"))

                    st.markdown(f"""
                    <div class="deal-card">
                        <div class="deal-title">Best Deal</div>
                        <div class="deal-medicine">
                            You selected {selected_medicine} (Rs {selected_price:.0f})
                        </div>
                        <div class="deal-detail">
                            Switch to <strong>{cheapest['name']}</strong> — Rs {cheapest['price_pkr']:.0f}
                            &nbsp;|&nbsp; <span class="deal-saving">Save Rs {saving:.0f}</span>
                        </div>
                        <div style="margin-top:0.5rem;">
                            <a class="wa-btn" href="{whatsapp_link(wa_text)}" target="_blank">Share on WhatsApp</a>
                            &nbsp;
                            <a href="{dawaai_url}" target="_blank" style="color:{COLORS['accent']}; font-size:0.8rem;">Dawaai.pk</a>
                            &nbsp;|&nbsp;
                            <a href="{medstore_url}" target="_blank" style="color:{COLORS['accent']}; font-size:0.8rem;">MedStore.com.pk</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="deal-card">
                        <div class="deal-title">Best Deal</div>
                        <div class="deal-medicine">{selected_medicine} — Rs {selected_price:.0f}</div>
                        <div class="deal-detail">This is already the best price for <strong>{generic}</strong></div>
                    </div>
                    """, unsafe_allow_html=True)

                # b) All alternatives table with "Savings vs Selected" column
                if len(alternatives["name"].unique()) > 1:
                    alt_display = alternatives[
                        ["name", "brand", "price_pkr", "drap_price_pkr",
                         "overprice_pct", "availability", "source"]
                    ].copy().sort_values("price_pkr")
                    alt_display["savings_vs_selected"] = selected_price - alt_display["price_pkr"]
                    alt_display.columns = [
                        "Medicine", "Brand", "Price (PKR)", "DRAP Price (PKR)",
                        "Overprice %", "Availability", "Source", "Savings vs Selected",
                    ]
                    st.dataframe(alt_display, width="stretch", hide_index=True)

                    # c) Price comparison bar chart (compact)
                    fig_alt = px.bar(
                        alternatives,
                        x="name",
                        y="price_pkr",
                        color="source",
                        barmode="group",
                        labels={"name": "Medicine", "price_pkr": "Price (PKR)", "source": "Source"},
                        title=f"Price Comparison — {generic}",
                        color_discrete_sequence=[COLORS["blue"], COLORS["green"]],
                    )
                    drap_val = alternatives["drap_price_pkr"].iloc[0]
                    if drap_val and drap_val > 0:
                        fig_alt.add_hline(
                            y=drap_val,
                            line_dash="dot",
                            line_color=COLORS["green_light"],
                            annotation_text=f"DRAP: Rs {drap_val:.0f}",
                            annotation_position="top left",
                            annotation_font_size=10,
                            annotation_font_color=COLORS["green_light"],
                        )
                    fig_alt.update_layout(**PLOTLY_LAYOUT, height=350)
                    st.plotly_chart(fig_alt, width="stretch")
                else:
                    st.info(f"No other brands found for **{generic}**.")
            else:
                st.info("No generic name information available for this medicine.")

    # =================================================================
    # 2. PRESCRIPTION CART
    # =================================================================
    with tab_rx:
        st.caption("Add all medicines from your prescription to compare total cost across pharmacies.")

        # --- Prescription Scanner ---
        drap_names, generic_map = load_drap_lookup()

        with st.expander("Scan Prescription", expanded=False):
            tab_upload, tab_paste = st.tabs(["Upload / Camera", "Type / Paste"])

            with tab_upload:
                uploaded_file = st.file_uploader(
                    "Upload prescription image",
                    type=["png", "jpg", "jpeg"],
                    key="_rx_upload",
                )
                camera_input = st.camera_input("Or take a photo", key="_rx_camera")

                rx_image = uploaded_file or camera_input
                if rx_image:
                    st.image(rx_image, caption="Prescription image", width=300)

                    # Check for Gemini API key: secrets > env > user input
                    try:
                        gemini_key = st.secrets["GEMINI_API_KEY"].strip()
                    except (KeyError, FileNotFoundError, AttributeError):
                        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
                    if not gemini_key:
                        gemini_key = st.text_input(
                            "Gemini API Key",
                            type="password",
                            key="_gemini_key_input",
                            help="Get a free key at https://aistudio.google.com/apikey",
                        )

                    if gemini_key:
                        import hashlib
                        img_hash = hashlib.md5(rx_image.getvalue()).hexdigest()
                        if st.session_state.get("_rx_img_hash") != img_hash:
                            with st.spinner("Reading prescription with Gemini..."):
                                try:
                                    import base64, requests as _req
                                    rx_image.seek(0)
                                    img_bytes = rx_image.getvalue()
                                    img_b64 = base64.b64encode(img_bytes).decode()
                                    prompt_text = (
                                        "Read this prescription image. "
                                        "Extract ONLY the medicine names, one per line. "
                                        "Include dosage if visible (e.g. 500mg). "
                                        "Do not add any other text, headers, or explanations."
                                    )
                                    payload = {
                                        "contents": [{
                                            "parts": [
                                                {"text": prompt_text},
                                                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                                            ]
                                        }]
                                    }

                                    ocr_result = None
                                    last_error = None
                                    for model_name in ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]:
                                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
                                        resp = _req.post(url, json=payload, timeout=30)
                                        if resp.status_code == 200:
                                            data = resp.json()
                                            ocr_result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                                            break
                                        else:
                                            last_error = resp.text
                                            continue

                                    if ocr_result:
                                        st.session_state["_rx_ocr_text"] = ocr_result
                                        st.session_state["_rx_img_hash"] = img_hash
                                    else:
                                        raise Exception(f"All models failed. Last error: {last_error}")
                                except Exception as e:
                                    err_msg = str(e)
                                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                                        st.error("Gemini free tier quota exhausted. Please wait a few minutes or check your plan at https://ai.google.dev/gemini-api/docs/rate-limits")
                                    else:
                                        st.error(f"Gemini API error: {err_msg}")

                        ocr_text = st.session_state.get("_rx_ocr_text", "")
                        if ocr_text:
                            st.text_area(
                                "Extracted medicines (edit if needed)",
                                value=ocr_text,
                                height=150,
                                key="_rx_ocr_edit",
                            )
                    else:
                        st.caption(
                            "Enter a [free Gemini API key](https://aistudio.google.com/apikey) "
                            "above to auto-extract medicine names from the image."
                        )

            with tab_paste:
                st.text_area(
                    "Enter medicine names (one per line)",
                    placeholder="Panadol 500mg\nDiane 35\nBrufen 400mg",
                    height=150,
                    key="_rx_paste",
                    help="Type medicine names as written on your prescription. "
                         "Fuzzy matching will find the closest match even with typos.",
                )

            # Determine the text to use for matching
            scan_text = (
                st.session_state.get("_rx_ocr_edit", "").strip()
                or st.session_state.get("_rx_ocr_text", "").strip()
                or st.session_state.get("_rx_paste", "").strip()
            )

            if st.button("Find Medicines", key="_rx_find_btn"):
                if scan_text:
                    matches = extract_medicines_from_text(scan_text, drap_names, generic_map)
                    st.session_state["scanner_matches"] = matches
                    if not matches:
                        st.warning("No medicines could be matched. Try different text.")
                else:
                    st.warning(
                        "The text box is empty. Click inside the text area above, "
                        "type the medicine names (one per line), then click **Find Medicines**."
                    )

            # Display match results
            scanner_matches = st.session_state.get("scanner_matches", [])
            if scanner_matches:
                selected_meds = {}
                for i, entry in enumerate(scanner_matches):
                    top_match = entry["matches"][0]
                    default_checked = top_match["score"] >= 0.6
                    col_check, col_info = st.columns([0.05, 0.95])
                    with col_check:
                        checked = st.checkbox(
                            "sel",
                            value=default_checked,
                            key=f"_rx_sel_{i}",
                            label_visibility="collapsed",
                        )
                    with col_info:
                        match_label = (
                            f"**{top_match['name']}** — "
                            f"_{top_match['match_type']}_ (score: {top_match['score']:.0%})"
                        )
                        st.markdown(
                            f'<div class="scanner-match">'
                            f'<span class="scanner-match-name">{top_match["name"]}</span>'
                            f'<span class="scanner-match-detail">'
                            f'{top_match["match_type"]} &middot; {top_match["score"]:.0%} '
                            f'&middot; from: "{entry["line"]}"'
                            f'</span></div>',
                            unsafe_allow_html=True,
                        )
                        # Show alternatives in a popover if available
                        if len(entry["matches"]) > 1:
                            with st.popover("Alternatives"):
                                for alt in entry["matches"][1:]:
                                    alt_checked = st.checkbox(
                                        f"{alt['name']} ({alt['match_type']}, {alt['score']:.0%})",
                                        value=False,
                                        key=f"_rx_alt_{i}_{alt['name']}",
                                    )
                                    if alt_checked:
                                        selected_meds[alt["name"]] = True
                    if checked:
                        selected_meds[top_match["name"]] = True

                meds_to_add = [m for m in selected_meds if m in all_medicines]
                if meds_to_add:
                    if st.button(
                        f"Add {len(meds_to_add)} medicine(s) to cart",
                        key="_rx_add_btn",
                        type="primary",
                    ):
                        existing = st.session_state.get("prescription_cart", [])
                        merged = list(dict.fromkeys(existing + meds_to_add))
                        st.session_state["prescription_cart"] = merged
                        st.session_state["scanner_matches"] = []
                        st.rerun()
                else:
                    st.info("No matched medicines found in the database. Try different text.")
        # --- End Scanner ---

        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Your Cart</div>', unsafe_allow_html=True)

        cart_medicines = st.multiselect(
            "Add medicines to your prescription",
            options=all_medicines,
            key="prescription_cart",
            label_visibility="collapsed",
        )

        if cart_medicines:
            cart_data = df[df["name"].isin(cart_medicines)].sort_values("scraped_at").drop_duplicates(
                subset=["name", "source"], keep="last"
            )
            source_totals = cart_data.groupby("source").agg(
                total=("price_pkr", "sum"),
                available=("availability", lambda s: (s == "In Stock").sum()),
                count=("name", "nunique"),
            ).reset_index()
            source_totals = source_totals.sort_values("total")

            if not source_totals.empty:
                cheapest_source = source_totals.iloc[0]
                most_exp_source = source_totals.iloc[-1] if len(source_totals) > 1 else None
                cart_saving = (most_exp_source["total"] - cheapest_source["total"]) if most_exp_source is not None else 0

                cart_cols = st.columns(len(source_totals))
                for i, (_, row) in enumerate(source_totals.iterrows()):
                    is_cheapest = i == 0 and len(source_totals) > 1
                    card_cls = "cart-summary cart-winner" if is_cheapest else "cart-summary"
                    badge = f'<span class="pill pill-green" style="margin-left:8px;">CHEAPEST</span>' if is_cheapest else ""
                    with cart_cols[i]:
                        saving_html = f' &nbsp;|&nbsp; <span class="deal-saving">Save Rs {cart_saving:,.0f}</span>' if is_cheapest and cart_saving > 0 else ''
                        color = COLORS['green'] if is_cheapest else '#ffffff'
                        card_html = f'<div class="{card_cls}"><div class="metric-label">{row["source"]}{badge}</div><div class="cart-total" style="color:{color};">Rs {row["total"]:,.0f}</div><div class="metric-sub">{int(row["available"])} / {int(row["count"])} in stock{saving_html}</div></div>'
                        st.markdown(card_html, unsafe_allow_html=True)

                cart_detail = cart_data[["name", "price_pkr", "availability", "source"]].copy()
                cart_detail.columns = ["Medicine", "Price (PKR)", "Availability", "Source"]
                st.dataframe(cart_detail, width="stretch", hide_index=True)

                cart_lines = [f"My Prescription Cost ({len(cart_medicines)} medicines):"]
                for _, row in source_totals.iterrows():
                    cart_lines.append(f"  {row['source']}: Rs {row['total']:,.0f}")
                if cart_saving > 0:
                    cart_lines.append(f"Save Rs {cart_saving:,.0f} by buying from {cheapest_source['source']}!")
                cart_lines.append("Tracked on MedTracker PK")
                wa_cart_url = whatsapp_link("\n".join(cart_lines))
                st.markdown(
                    f'<a class="wa-btn" href="{wa_cart_url}" target="_blank">Share prescription cost on WhatsApp</a>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Select medicines above to see total prescription cost across pharmacies.")

    # =================================================================
    # 3. MONTHLY COST CALCULATOR
    # =================================================================
    with tab_calc:
        st.caption("For patients who buy the same medicines regularly — see projected costs and generic savings.")

        calc_col1, calc_col2, calc_col3 = st.columns([3, 1, 1])
        with calc_col1:
            calc_med = st.selectbox("Medicine", options=all_medicines, key="cost_calc_med")
        with calc_col2:
            qty_per_month = st.number_input("Qty / month", min_value=1, max_value=100, value=1, key="cost_calc_qty")
        with calc_col3:
            num_months = st.number_input("Months", min_value=1, max_value=24, value=6, key="cost_calc_months")

        if calc_med:
            calc_data = df[df["name"] == calc_med].sort_values("scraped_at").drop_duplicates(
                subset=["source"], keep="last"
            )

            if not calc_data.empty:
                cheapest = calc_data.sort_values("price_pkr").iloc[0]
                unit_price = cheapest["price_pkr"]
                total_cost = unit_price * qty_per_month * num_months
                source_name = cheapest["source"]
                generic = cheapest.get("generic_name", "")

                generic_saving_html = ""
                if generic:
                    all_generic = df[
                        df["generic_name"].str.lower() == generic.lower()
                    ].sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")
                    cheapest_generic = all_generic.sort_values("price_pkr").iloc[0]

                    if cheapest_generic["name"] != calc_med and cheapest_generic["price_pkr"] < unit_price:
                        alt_total = cheapest_generic["price_pkr"] * qty_per_month * num_months
                        saved = total_cost - alt_total
                        generic_saving_html = f"""
                        <div style="margin-top:0.75rem; padding:0.75rem 1rem; background:{COLORS['green_dim']}; border:1px solid rgba(255,255,255,0.1); border-radius:8px;">
                            <span style="color:{COLORS['green']}; font-weight:600;">Generic alternative:</span>
                            <span style="color:{COLORS['text']};">
                                Switch to <strong>{cheapest_generic['name']}</strong> (same {generic}) —
                                Rs {cheapest_generic['price_pkr']:.0f}/unit from {cheapest_generic['source']}
                                &nbsp;|&nbsp; <strong style="color:{COLORS['green']};">Save Rs {saved:,.0f}</strong> over {num_months} months
                            </span>
                        </div>
                        """

                st.markdown(f"""
                <div class="deal-card">
                    <div class="deal-title">Projected Cost</div>
                    <div class="deal-medicine">Rs {total_cost:,.0f}</div>
                    <div class="deal-detail">
                        {calc_med} — Rs {unit_price:.0f}/unit x {qty_per_month}/month x {num_months} months
                        &nbsp;|&nbsp; Best at <strong>{source_name}</strong>
                    </div>
                    {generic_saving_html}
                </div>
                """, unsafe_allow_html=True)

                wa_cost_text = (
                    f"Monthly medicine cost: {calc_med}\n"
                    f"Rs {unit_price:.0f}/unit x {qty_per_month}/mo x {num_months} mo = Rs {total_cost:,.0f}\n"
                    f"Best price at {source_name}\n"
                    f"Tracked on MedTracker PK"
                )
                st.markdown(
                    f'<a class="wa-btn" href="{whatsapp_link(wa_cost_text)}" target="_blank">Share cost breakdown on WhatsApp</a>',
                    unsafe_allow_html=True,
                )

    # =================================================================
    # 4. PHARMACIES NEAR YOU
    # =================================================================
    with tab_pharm:
        user_city = st.session_state.get("user_city", None)
        if user_city:
            st.caption(f"Showing pharmacies near {user_city}")
        else:
            st.caption("Find pharmacies near you across Pakistan.")

        pharmacies_df = load_pharmacies()
        if not pharmacies_df.empty:
            phy_df = pharmacies_df.copy()
            if selected_city != "All":
                phy_df = phy_df[phy_df["city"] == selected_city]

            # Map first for visual impact
            if selected_city != "All":
                center_lat = phy_df["latitude"].mean()
                center_lon = phy_df["longitude"].mean()
                zoom = 11
            else:
                center_lat = 30.3753
                center_lon = 69.3451
                zoom = 5

            fig_map = px.scatter_mapbox(
                phy_df,
                lat="latitude",
                lon="longitude",
                hover_name="name",
                hover_data={"address": True, "city": True, "phone": True, "latitude": False, "longitude": False},
                color_discrete_sequence=[COLORS["green"]],
                zoom=zoom,
                center={"lat": center_lat, "lon": center_lon},
                height=450,
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_map, width="stretch")

            phy_display = phy_df[["name", "address", "city", "phone"]].copy()
            phy_display.columns = ["Pharmacy", "Address", "City", "Phone"]
            st.dataframe(phy_display, width="stretch", hide_index=True)

            # Online links
            search_term = selected_medicine if selected_medicine else ""
            if search_term:
                dawaai_url = DAWAAI_SEARCH_URL.format(query=search_term.replace(" ", "+"))
                medstore_url = MEDSTORE_SEARCH_URL.format(query=search_term.replace(" ", "+"))
                st.caption(
                    f'Also available online: '
                    f'<a href="{dawaai_url}" target="_blank">Dawaai.pk</a> | '
                    f'<a href="{medstore_url}" target="_blank">MedStore.com.pk</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption(
                    'Also available online: '
                    '<a href="https://dawaai.pk" target="_blank">Dawaai.pk</a> | '
                    '<a href="https://medstore.com.pk" target="_blank">MedStore.com.pk</a>',
                    unsafe_allow_html=True,
                )

            # WhatsApp share
            phy_lines = [f"Pharmacies in {selected_city if selected_city != 'All' else 'Pakistan'}:"]
            for _, p in phy_df.head(5).iterrows():
                phy_lines.append(f"  {p['name']} — {p['address']}, {p['city']} ({p['phone']})")
            phy_lines.append("Found on MedTracker PK")
            st.markdown(
                f'<a class="wa-btn" href="{whatsapp_link(chr(10).join(phy_lines))}" target="_blank">Share pharmacy list on WhatsApp</a>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Pharmacy location data not available.")

    # =================================================================
    # 5. ANALYTICS
    # =================================================================
    with tab_analytics:
        # Inline filters (moved from sidebar)
        flt_col1, flt_col2, flt_col3 = st.columns(3)
        with flt_col1:
            overpriced_only = st.toggle("Overpriced only", value=False, key="analytics_overpriced")
        with flt_col2:
            sources = ["All"] + sorted(df["source"].unique().tolist()) if not df.empty else ["All"]
            selected_source = st.selectbox("Source", sources, key="analytics_source")
        with flt_col3:
            avail_options = ["All", "In Stock", "Out of Stock", "Limited", "Unknown"]
            selected_avail = st.selectbox("Availability", avail_options, key="analytics_avail")

        filtered = df.copy()
        if overpriced_only:
            filtered = filtered[filtered["overpriced"] == 1]
        if selected_source != "All":
            filtered = filtered[filtered["source"] == selected_source]
        if selected_avail != "All":
            filtered = filtered[filtered["availability"] == selected_avail]

        # Compact metrics row (3 cards)
        latest = df.sort_values("scraped_at").drop_duplicates(subset=["name"], keep="last")
        overpriced_n = int(latest["overpriced"].sum())
        last_updated = df["scraped_at"].max()
        try:
            ts = datetime.fromisoformat(last_updated).strftime("%d %b %Y, %I:%M %p")
        except Exception:
            ts = str(last_updated)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Tracked</div>
                <div class="metric-value">{len(latest)}</div>
                <div class="metric-sub">unique medicines</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Overpriced</div>
                <div class="metric-value blue">{overpriced_n}</div>
                <div class="metric-sub">above DRAP threshold</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last Updated</div>
                <div class="metric-value" style="font-size:1.1rem; margin-top:0.25rem;">{ts}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Full data table
        st.markdown('<div class="section-title">Price Comparison</div>', unsafe_allow_html=True)

        if filtered.empty:
            st.info("No medicines match your current filters.")
        else:
            display_cols = [
                "name", "brand", "price_pkr", "drap_price_pkr",
                "overprice_pct", "overpriced", "availability", "source", "scraped_at",
            ]
            display_df = filtered[display_cols].copy()
            display_df["overpriced"] = display_df["overpriced"].map(
                {1: "OVERPRICED", 0: "FAIR"}
            )
            display_df.columns = [
                "Medicine", "Brand", "Price (PKR)", "DRAP Price (PKR)",
                "Overprice %", "Status", "Availability", "Source", "Scraped At",
            ]

            st.dataframe(display_df, width="stretch", hide_index=True, height=400)

            # Availability pills
            avail_counts = filtered["availability"].value_counts()
            pills_html = " &nbsp; ".join(
                f'<span class="pill {cls}">{label}: {avail_counts.get(label, 0)}</span>'
                for label, cls in [
                    ("In Stock", "pill-green"),
                    ("Limited", "pill-lblue"),
                    ("Out of Stock", "pill-blue"),
                    ("Unknown", "pill-gray"),
                ]
            )
            st.markdown(f'<div style="margin: 0.5rem 0 1rem;">{pills_html}</div>', unsafe_allow_html=True)

            # Export
            exp1, exp2, _ = st.columns([1, 1, 4])
            with exp1:
                csv_data = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"medicine_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )
            with exp2:
                report_lines = [
                    "Pakistan Medicine Price Tracker - Report",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    f"Total Medicines: {len(display_df)}",
                    f"Overpriced: {(display_df['Status'] == 'OVERPRICED').sum()}",
                    f"Fairly Priced: {(display_df['Status'] == 'FAIR').sum()}",
                    "",
                    "--- Detailed Data ---",
                    "",
                ]
                report_text = "\n".join(report_lines) + "\n" + display_df.to_csv(index=False)
                st.download_button(
                    label="Download Report",
                    data=report_text.encode("utf-8"),
                    file_name=f"medicine_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                )

        # Overpricing Analysis
        st.markdown('<div class="section-title">Overpricing Analysis</div>', unsafe_allow_html=True)

        latest_filtered = filtered.sort_values("scraped_at").drop_duplicates(
            subset=["name"], keep="last"
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            top_overpriced = (
                latest_filtered[latest_filtered["overprice_pct"] > 0]
                .nlargest(10, "overprice_pct")
            )
            if not top_overpriced.empty:
                fig_bar = px.bar(
                    top_overpriced,
                    x="overprice_pct",
                    y="name",
                    orientation="h",
                    color="overprice_pct",
                    color_continuous_scale=["#52525b", "#fafafa"],
                    labels={"overprice_pct": "% Above DRAP", "name": ""},
                    title="Top 10 Overpriced",
                )
                bar_layout = {**PLOTLY_LAYOUT}
                bar_layout["yaxis"] = dict(autorange="reversed", gridcolor=COLORS["border"])
                fig_bar.update_layout(
                    **bar_layout,
                    showlegend=False,
                    height=400,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, width="stretch")
            else:
                st.info("No overpriced medicines to display.")

        with chart_col2:
            if not latest_filtered.empty:
                overpriced_count = int(latest_filtered["overpriced"].sum())
                fair_count = len(latest_filtered) - overpriced_count

                fig_pie = go.Figure(data=[go.Pie(
                    labels=["Overpriced", "Fairly Priced"],
                    values=[overpriced_count, fair_count],
                    hole=0.55,
                    marker=dict(colors=["#a1a1aa", "#3f3f46"]),
                    textfont=dict(color="#ffffff", size=13),
                    hoverinfo="label+value+percent",
                )])
                fig_pie.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Price Distribution",
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5,
                        font=dict(color=COLORS["text"]),
                    ),
                )
                st.plotly_chart(fig_pie, width="stretch")
            else:
                st.info("No data to display.")

        # Availability Chart
        st.markdown('<div class="section-title">Availability Across Pharmacies</div>', unsafe_allow_html=True)

        if "availability" in filtered.columns and not latest_filtered.empty:
            avail_data = filtered.sort_values("scraped_at").drop_duplicates(
                subset=["name", "source"], keep="last"
            )
            avail_cross = avail_data.groupby(["name", "source"])["availability"].first().reset_index()

            color_map = {
                "In Stock": "#fafafa",
                "Limited": "#a1a1aa",
                "Out of Stock": "#52525b",
                "Unknown": COLORS["text_dim"],
            }

            fig_avail = px.bar(
                avail_cross,
                x="name",
                y=avail_cross["availability"].map(
                    {"In Stock": 1, "Limited": 0.5, "Out of Stock": 0, "Unknown": 0.25}
                ),
                color="availability",
                barmode="group",
                facet_col="source",
                color_discrete_map=color_map,
                labels={"name": "Medicine", "y": "Stock Level", "availability": "Status"},
                title="Medicine Availability by Pharmacy",
                category_orders={"availability": ["In Stock", "Limited", "Out of Stock", "Unknown"]},
            )
            fig_avail.update_layout(
                **PLOTLY_LAYOUT,
                height=420,
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.35,
                    xanchor="center", x=0.5,
                ),
            )
            fig_avail.update_yaxes(
                tickvals=[0, 0.25, 0.5, 1],
                ticktext=["Out of Stock", "Unknown", "Limited", "In Stock"],
            )
            st.plotly_chart(fig_avail, width="stretch")
        else:
            st.info("No availability data to display.")

        # Price Trends
        st.markdown('<div class="section-title">Price Trends</div>', unsafe_allow_html=True)

        med_counts = filtered.groupby("name").size()
        trending_meds = med_counts[med_counts > 1].index.tolist()

        if trending_meds:
            default_meds = trending_meds[:5]
            selected_meds = st.multiselect(
                "Select medicines to compare",
                options=trending_meds,
                default=default_meds,
                label_visibility="collapsed",
            )

            if selected_meds:
                trend_data = filtered[filtered["name"].isin(selected_meds)].copy()
                trend_data["scraped_at"] = pd.to_datetime(trend_data["scraped_at"])
                trend_data = trend_data.sort_values("scraped_at")

                fig_trend = px.line(
                    trend_data,
                    x="scraped_at",
                    y="price_pkr",
                    color="name",
                    markers=True,
                    labels={
                        "scraped_at": "Date",
                        "price_pkr": "Price (PKR)",
                        "name": "Medicine",
                    },
                    title="Medicine Price Trends",
                    color_discrete_sequence=[
                        "#fafafa", "#a1a1aa", "#d4d4d8",
                        "#71717a", "#e4e4e7", "#52525b", "#b4b4b8",
                        "#8a8a8f", "#636366", "#3f3f46",
                    ],
                )

                for med_name in selected_meds:
                    med_data = trend_data[trend_data["name"] == med_name]
                    if not med_data.empty:
                        drap_val = med_data["drap_price_pkr"].iloc[0]
                        if drap_val and drap_val > 0:
                            fig_trend.add_hline(
                                y=drap_val,
                                line_dash="dot",
                                line_color=COLORS["text_dim"],
                                opacity=0.3,
                                annotation_text=f"DRAP: {med_name}",
                                annotation_position="top left",
                                annotation_font_size=9,
                                annotation_font_color=COLORS["text_dim"],
                            )

                if "anomaly" in trend_data.columns:
                    anomalies = trend_data[trend_data["anomaly"] == 1]
                    if not anomalies.empty:
                        fig_trend.add_trace(
                            go.Scatter(
                                x=anomalies["scraped_at"],
                                y=anomalies["price_pkr"],
                                mode="markers",
                                marker=dict(
                                    symbol="diamond",
                                    size=12,
                                    color=COLORS["blue"],
                                    line=dict(width=1, color="#ffffff"),
                                ),
                                name="Anomaly",
                                hovertext=anomalies["name"] + " (anomaly)",
                            )
                        )

                fig_trend.update_layout(
                    **PLOTLY_LAYOUT,
                    height=450,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.25,
                        xanchor="center",
                        x=0.5,
                    ),
                )
                st.plotly_chart(fig_trend, width="stretch")
            else:
                st.info("Select at least one medicine to see trends.")
        else:
            st.info("Not enough data points for trend analysis. Run the scraper multiple times to build history.")

    # =================================================================
    # 6. RESEARCH INSIGHTS
    # =================================================================
    with tab_research:
        from app.research_insights import render_research_insights
        render_research_insights(df, load_pharmacies(), COLORS, PLOTLY_LAYOUT)

    # --- Footer ---
    st.markdown(f"""
    <div class="footer">
        Data sourced from <strong>Dawaai.pk</strong> & <strong>MedStore.com.pk</strong>
        &nbsp;&middot;&nbsp;
        Compared against <strong>DRAP</strong> registered prices
        &nbsp;&middot;&nbsp;
        Built by <a href="https://github.com/rushammm">Rusham Elahi</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
