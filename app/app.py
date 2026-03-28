"""
Pakistan Medicine Price Tracker — Streamlit Dashboard
=====================================================
Run with:  streamlit run app/app.py
"""

import os
import sys
import sqlite3
from datetime import datetime

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
# Color palette  —  green & blue only
# ---------------------------------------------------------------------------

COLORS = {
    "bg":         "#0a0a0a",
    "surface":    "#111111",
    "surface2":   "#1a1a1a",
    "border":     "#222222",
    "border_l":   "#2a2a2a",
    "text":       "#e0e0e0",
    "text_dim":   "#666666",
    "accent":     "#3b82f6",      # blue-500
    "accent_dim": "#1d4ed8",      # blue-700
    "blue":       "#3b82f6",      # blue-500
    "blue_dim":   "#1e3a5f",      # dark blue
    "blue_light": "#60a5fa",      # blue-400
    "green":      "#22c55e",      # green-500
    "green_dim":  "#14532d",      # dark green
    "green_light":"#34d399",      # green-400
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

    /* Header */
    .hero {{
        background: {COLORS["surface"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }}
    .hero::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS["blue"]}, {COLORS["green"]}, {COLORS["blue"]});
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
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        transition: border-color 0.2s;
    }}
    .metric-card:hover {{
        border-color: {COLORS["border_l"]};
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
        border: 1px solid {COLORS["blue"]}33;
    }}
    .pill-lblue {{
        background: {COLORS["blue_dim"]}44;
        color: {COLORS["blue_light"]};
        border: 1px solid {COLORS["blue_light"]}33;
    }}
    .pill-green {{
        background: {COLORS["green_dim"]};
        color: {COLORS["green"]};
        border: 1px solid {COLORS["green"]}33;
    }}
    .pill-gray {{
        background: #33333344;
        color: {COLORS["text_dim"]};
        border: 1px solid {COLORS["text_dim"]}33;
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
        background: {COLORS["surface"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {COLORS["text"]} !important;
    }}

    /* Plotly chart backgrounds */
    .stPlotlyChart {{
        background: {COLORS["surface"]};
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
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
        color: {COLORS["text"]} !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: {COLORS["accent"]} !important;
        box-shadow: 0 0 0 1px {COLORS["accent"]}44 !important;
    }}

    /* Multiselect */
    .stMultiSelect > div > div {{
        background: {COLORS["surface2"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        border-radius: 8px !important;
    }}

    /* Download buttons */
    .stDownloadButton > button {{
        background: {COLORS["surface2"]} !important;
        border: 1px solid {COLORS["border"]} !important;
        color: {COLORS["text"]} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}
    .stDownloadButton > button:hover {{
        border-color: {COLORS["accent"]} !important;
        color: #ffffff !important;
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
        with st.spinner("Initializing data..."):
            from scraper.scrape import run_scraper
            run_scraper()
        st.cache_data.clear()


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
    st.sidebar.markdown("### Filters")

    if st.sidebar.button("Refresh Data", width="stretch"):
        with st.spinner("Scraping prices..."):
            from scraper.scrape import run_scraper
            run_scraper()
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("")

    overpriced_only = st.sidebar.toggle("Overpriced only", value=False)

    sources = ["All"] + sorted(df["source"].unique().tolist()) if not df.empty else ["All"]
    selected_source = st.sidebar.selectbox("Source", sources)

    avail_options = ["All", "In Stock", "Out of Stock", "Limited", "Unknown"]
    selected_avail = st.sidebar.selectbox("Availability", avail_options)

    pharmacies_df = load_pharmacies()
    cities = ["All"] + sorted(pharmacies_df["city"].unique().tolist()) if not pharmacies_df.empty else ["All"]
    user_city = st.session_state.get("user_city", "All")
    default_idx = cities.index(user_city) if user_city in cities else 0
    selected_city = st.sidebar.selectbox("Pharmacy City", cities, index=default_idx)

    st.sidebar.markdown("---")
    st.sidebar.caption("Built by [Rusham Elahi](https://github.com/rushammm)")

    return overpriced_only, selected_source, selected_avail, selected_city


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_data()
    df = load_data()

    # --- Location popup (first visit only) ---
    if "user_city" not in st.session_state:
        location_dialog()

    # --- Hero ---
    st.markdown("""
    <div class="hero">
        <h1>MedTracker PK</h1>
        <p>Real-time pharmacy price monitoring vs DRAP regulated rates</p>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available. Click **Refresh Data** in the sidebar.")
        render_sidebar(df)
        return

    df = detect_anomalies_df(df)
    if "availability" not in df.columns:
        df["availability"] = "Unknown"
    overpriced_only, selected_source, selected_avail, selected_city = render_sidebar(df)

    filtered = df.copy()
    if overpriced_only:
        filtered = filtered[filtered["overpriced"] == 1]
    if selected_source != "All":
        filtered = filtered[filtered["source"] == selected_source]
    if selected_avail != "All":
        filtered = filtered[filtered["availability"] == selected_avail]

    # --- Metrics ---
    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name"], keep="last")
    overpriced_n = int(latest["overpriced"].sum())
    anomaly_n = int(df["anomaly"].sum()) if "anomaly" in df.columns else 0
    last_updated = df["scraped_at"].max()
    try:
        ts = datetime.fromisoformat(last_updated).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        ts = str(last_updated)

    avail_total = len(latest)
    avail_in_stock = int((latest["availability"] == "In Stock").sum()) if "availability" in latest.columns else 0

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
            <div class="metric-label">Anomalies</div>
            <div class="metric-value lblue">{anomaly_n}</div>
            <div class="metric-sub">unusual price patterns</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Availability</div>
            <div class="metric-value green">{avail_in_stock} / {avail_total}</div>
            <div class="metric-sub">in stock</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Last Updated</div>
            <div class="metric-value" style="font-size:1.1rem; margin-top:0.25rem;">{ts}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Search ---
    search = st.text_input(
        "Search medicines",
        placeholder="Type a medicine name (e.g. Panadol, Augmentin) ...",
        label_visibility="collapsed",
    )
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
            | filtered["generic_name"].str.contains(search, case=False, na=False)
        ]

    # --- Data table ---
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

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=400,
        )

        # Availability pills legend
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

    # --- Similar Formula Finder ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Similar Formula Finder</div>', unsafe_allow_html=True)

    all_medicines = df["name"].unique().tolist()
    selected_medicine = st.selectbox(
        "Select a medicine to find alternatives with the same active ingredient",
        options=sorted(all_medicines),
        key="formula_finder",
    )

    if selected_medicine:
        med_row = df[df["name"] == selected_medicine].iloc[0]
        generic = med_row.get("generic_name", "")

        if generic:
            st.markdown(
                f'Active ingredient: <span class="pill pill-green">{generic}</span>',
                unsafe_allow_html=True,
            )

            alternatives = df[
                df["generic_name"].str.lower() == generic.lower()
            ].sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")

            if len(alternatives["name"].unique()) > 1:
                alt_display = alternatives[
                    ["name", "brand", "price_pkr", "drap_price_pkr",
                     "overprice_pct", "availability", "source"]
                ].copy().sort_values("price_pkr")
                alt_display.columns = [
                    "Medicine", "Brand", "Price (PKR)", "DRAP Price (PKR)",
                    "Overprice %", "Availability", "Source",
                ]
                st.dataframe(alt_display, width="stretch", hide_index=True)

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
                fig_alt.update_layout(**PLOTLY_LAYOUT, height=400)
                st.plotly_chart(fig_alt, width="stretch")
            else:
                st.info(f"No other brands found for **{generic}**. This is the only tracked medicine with this formula.")
        else:
            st.info("No generic name information available for this medicine.")

    # --- Charts ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
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
                color_continuous_scale=[COLORS["blue_light"], COLORS["blue"]],
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
                marker=dict(colors=[COLORS["blue"], COLORS["green"]]),
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

    # --- Availability Chart ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Availability Across Pharmacies</div>', unsafe_allow_html=True)

    if "availability" in filtered.columns and not latest_filtered.empty:
        # Get latest record per medicine per source
        avail_data = filtered.sort_values("scraped_at").drop_duplicates(
            subset=["name", "source"], keep="last"
        )
        avail_cross = avail_data.groupby(["name", "source"])["availability"].first().reset_index()

        color_map = {
            "In Stock": COLORS["green"],
            "Limited": COLORS["blue_light"],
            "Out of Stock": COLORS["blue"],
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

    # --- Trends ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
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
                    COLORS["blue"], COLORS["green"], COLORS["blue_light"],
                    COLORS["green_light"], "#1d4ed8", "#15803d", "#93c5fd",
                    "#86efac", "#2563eb", "#16a34a",
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

    # --- Find a Pharmacy ---
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Find a Pharmacy</div>', unsafe_allow_html=True)

    # Part A — Online Pharmacies
    search_term = selected_medicine if selected_medicine else (search if search else "")
    dawaai_url = DAWAAI_SEARCH_URL.format(query=search_term.replace(" ", "+")) if search_term else "#"
    medstore_url = MEDSTORE_SEARCH_URL.format(query=search_term.replace(" ", "+")) if search_term else "#"

    online_col1, online_col2 = st.columns(2)

    if search_term:
        med_data_dawaai = df[
            (df["name"] == search_term) & (df["source"].str.contains("dawaai", case=False))
        ]
        med_data_medstore = df[
            (df["name"] == search_term) & (df["source"].str.contains("medstore", case=False))
        ]

        dawaai_avail = med_data_dawaai["availability"].iloc[0] if not med_data_dawaai.empty else "Unknown"
        dawaai_price = f"Rs {med_data_dawaai['price_pkr'].iloc[0]:.0f}" if not med_data_dawaai.empty else "N/A"
        medstore_avail = med_data_medstore["availability"].iloc[0] if not med_data_medstore.empty else "Unknown"
        medstore_price = f"Rs {med_data_medstore['price_pkr'].iloc[0]:.0f}" if not med_data_medstore.empty else "N/A"
    else:
        dawaai_avail = "Unknown"
        dawaai_price = "N/A"
        medstore_avail = "Unknown"
        medstore_price = "N/A"

    avail_pill_map = {
        "In Stock": "pill-green",
        "Limited": "pill-lblue",
        "Out of Stock": "pill-blue",
        "Unknown": "pill-gray",
    }

    with online_col1:
        pill_cls = avail_pill_map.get(dawaai_avail, "pill-gray")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Dawaai.pk</div>
            <span class="pill {pill_cls}">{dawaai_avail}</span>
            <div class="metric-value" style="font-size:1.3rem; margin-top:0.5rem;">{dawaai_price}</div>
            <div class="metric-sub" style="margin-top:0.5rem;">
                <a href="{dawaai_url}" target="_blank" style="color:{COLORS['accent']};">Search on Dawaai.pk</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with online_col2:
        pill_cls = avail_pill_map.get(medstore_avail, "pill-gray")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MedStore.com.pk</div>
            <span class="pill {pill_cls}">{medstore_avail}</span>
            <div class="metric-value" style="font-size:1.3rem; margin-top:0.5rem;">{medstore_price}</div>
            <div class="metric-sub" style="margin-top:0.5rem;">
                <a href="{medstore_url}" target="_blank" style="color:{COLORS['accent']};">Search on MedStore.com.pk</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Part B — Physical Pharmacies
    user_city = st.session_state.get("user_city", None)
    if user_city:
        st.markdown(
            f'<div class="section-title" style="margin-top:1.5rem;">Pharmacies Near You — '
            f'<span class="pill pill-green">{user_city}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="section-title" style="margin-top:1.5rem;">Physical Pharmacies</div>',
            unsafe_allow_html=True,
        )

    pharmacies_df = load_pharmacies()
    if not pharmacies_df.empty:
        phy_df = pharmacies_df.copy()
        if selected_city != "All":
            phy_df = phy_df[phy_df["city"] == selected_city]

        phy_display = phy_df[["name", "address", "city", "phone"]].copy()
        phy_display.columns = ["Pharmacy", "Address", "City", "Phone"]
        st.dataframe(phy_display, width="stretch", hide_index=True)

        # Map
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
    else:
        st.info("Pharmacy location data not available.")

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
