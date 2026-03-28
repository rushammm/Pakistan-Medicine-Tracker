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
sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "bg":         "#0a0a0a",
    "surface":    "#111111",
    "surface2":   "#1a1a1a",
    "border":     "#222222",
    "border_l":   "#2a2a2a",
    "text":       "#e0e0e0",
    "text_dim":   "#666666",
    "accent":     "#6c63ff",
    "accent_dim": "#4a42cc",
    "red":        "#ef4444",
    "red_dim":    "#7f1d1d",
    "green":      "#22c55e",
    "green_dim":  "#14532d",
    "amber":      "#f59e0b",
    "cyan":       "#06b6d4",
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
        background: linear-gradient(90deg, {COLORS["accent"]}, {COLORS["cyan"]}, {COLORS["accent"]});
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
    .metric-value.red {{ color: {COLORS["red"]}; }}
    .metric-value.amber {{ color: {COLORS["amber"]}; }}
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
    .pill-red {{
        background: {COLORS["red_dim"]};
        color: {COLORS["red"]};
        border: 1px solid {COLORS["red"]}33;
    }}
    .pill-green {{
        background: {COLORS["green_dim"]};
        color: {COLORS["green"]};
        border: 1px solid {COLORS["green"]}33;
    }}
    .pill-amber {{
        background: #451a0344;
        color: {COLORS["amber"]};
        border: 1px solid {COLORS["amber"]}33;
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

    st.sidebar.markdown("---")
    st.sidebar.caption("Built by [Rusham Elahi](https://github.com/rushammm)")

    return overpriced_only, selected_source, selected_avail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_data()
    df = load_data()

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
    overpriced_only, selected_source, selected_avail = render_sidebar(df)

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
            <div class="metric-value red">{overpriced_n}</div>
            <div class="metric-sub">above DRAP threshold</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Anomalies</div>
            <div class="metric-value amber">{anomaly_n}</div>
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
                ("Limited", "pill-amber"),
                ("Out of Stock", "pill-red"),
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
                color_continuous_scale=[COLORS["amber"], COLORS["red"]],
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
                marker=dict(colors=[COLORS["red"], COLORS["green"]]),
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
            "Limited": COLORS["amber"],
            "Out of Stock": COLORS["red"],
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
                    COLORS["accent"], COLORS["cyan"], COLORS["amber"],
                    COLORS["green"], COLORS["red"], "#a78bfa", "#f472b6",
                    "#34d399", "#fbbf24", "#60a5fa",
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
                                color=COLORS["red"],
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
