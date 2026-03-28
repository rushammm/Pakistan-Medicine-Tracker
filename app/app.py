"""
Pakistan Medicine Price Tracker — Streamlit Dashboard
=====================================================
A professional dashboard that visualises medicine prices scraped from
Pakistani pharmacy websites and flags overpricing against DRAP rates.

Run with:  streamlit run app/app.py
"""

import os
import sys
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medicines.db")

# Add project root so we can import the scraper
sys.path.insert(0, BASE_DIR)


# ---------------------------------------------------------------------------
# Page configuration & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pakistan Medicine Price Tracker",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #0d6efd 0%, #198754 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 1rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
    }

    /* Overpriced badge */
    .badge-overpriced {
        background-color: #dc3545;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fair {
        background-color: #198754;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #dee2e6;
        color: #6c757d;
        font-size: 0.85rem;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    """Load medicine data from SQLite. Returns an empty DataFrame if DB is missing."""
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
    """
    Make sure the database exists and has data.
    If not, run the scraper automatically so the app always shows something.
    """
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
        with st.spinner("First run — generating medicine data..."):
            from scraper.scrape import run_scraper
            run_scraper()
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame):
    """Render the sidebar controls and return filter settings."""
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("---")

    # Run scraper button
    if st.sidebar.button("🔄 Run Scraper", use_container_width=True):
        with st.spinner("Scraping medicine prices..."):
            from scraper.scrape import run_scraper
            run_scraper()
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    # Filter: overpriced only
    overpriced_only = st.sidebar.toggle("🔴 Show overpriced only", value=False)

    # Filter: source
    sources = ["All"] + sorted(df["source"].unique().tolist()) if not df.empty else ["All"]
    selected_source = st.sidebar.selectbox("📦 Filter by source", sources)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Built by [Rusham Elahi](https://github.com/)**"
    )

    return overpriced_only, selected_source


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main():
    # Ensure we have data before anything else
    ensure_data()

    # Load data
    df = load_data()

    # ------ Header ------
    st.markdown("""
    <div class="main-header">
        <h1>💊 Pakistan Medicine Price Tracker</h1>
        <p>Monitoring pharmacy prices vs DRAP registered rates</p>
    </div>
    """, unsafe_allow_html=True)

    # Handle empty data edge case
    if df.empty:
        st.warning("No data available. Click **Run Scraper** in the sidebar to fetch prices.")
        render_sidebar(df)
        return

    # Sidebar filters
    overpriced_only, selected_source = render_sidebar(df)

    # Apply filters
    filtered = df.copy()
    if overpriced_only:
        filtered = filtered[filtered["overpriced"] == 1]
    if selected_source != "All":
        filtered = filtered[filtered["source"] == selected_source]

    # ------ Top metrics row ------
    # Use the latest record per medicine for aggregate stats
    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name"], keep="last")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Medicines Tracked", len(latest))
    with col2:
        overpriced_n = int(latest["overpriced"].sum())
        st.metric(
            "🔴 Overpriced",
            overpriced_n,
            delta=f"{overpriced_n} flagged" if overpriced_n > 0 else "None",
            delta_color="inverse",
        )
    with col3:
        last_updated = df["scraped_at"].max()
        try:
            ts = datetime.fromisoformat(last_updated).strftime("%d %b %Y, %I:%M %p")
        except Exception:
            ts = str(last_updated)
        st.metric("🕒 Last Updated", ts)

    st.markdown("---")

    # ------ Search bar ------
    search = st.text_input(
        "🔍 Search medicines",
        placeholder="Type a medicine name (e.g. Panadol, Augmentin) ...",
    )
    if search:
        filtered = filtered[
            filtered["name"].str.contains(search, case=False, na=False)
            | filtered["generic_name"].str.contains(search, case=False, na=False)
        ]

    # ------ Main data table ------
    st.subheader("Medicine Price Comparison")

    if filtered.empty:
        st.info("No medicines match your current filters.")
    else:
        # Prepare display dataframe
        display_cols = [
            "name", "brand", "price_pkr", "drap_price_pkr",
            "overprice_pct", "overpriced", "source", "scraped_at",
        ]
        display_df = filtered[display_cols].copy()
        display_df["overpriced"] = display_df["overpriced"].map(
            {1: "🔴 Overpriced", 0: "🟢 Fair"}
        )
        display_df.columns = [
            "Medicine", "Brand", "Price (PKR)", "DRAP Price (PKR)",
            "Overprice %", "Status", "Source", "Scraped At",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

    st.markdown("---")

    # ------ Charts row: bar + pie side by side ------
    st.subheader("📊 Overpricing Analysis")

    latest_filtered = filtered.sort_values("scraped_at").drop_duplicates(
        subset=["name"], keep="last"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Bar chart: Top 10 most overpriced medicines
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
                color_continuous_scale=["#ffc107", "#dc3545"],
                labels={"overprice_pct": "% Above DRAP", "name": "Medicine"},
                title="Top 10 Most Overpriced Medicines",
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No overpriced medicines to display.")

    with chart_col2:
        # Pie chart: Overpriced vs fairly priced
        if not latest_filtered.empty:
            overpriced_count = int(latest_filtered["overpriced"].sum())
            fair_count = len(latest_filtered) - overpriced_count

            fig_pie = px.pie(
                names=["Overpriced", "Fairly Priced"],
                values=[overpriced_count, fair_count],
                color=["Overpriced", "Fairly Priced"],
                color_discrete_map={
                    "Overpriced": "#dc3545",
                    "Fairly Priced": "#198754",
                },
                title="Overpriced vs Fairly Priced",
                hole=0.4,
            )
            fig_pie.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data to display.")

    st.markdown("---")

    # ------ Price trend line chart ------
    st.subheader("📈 Price Trends Over Time")

    # Get medicines that have more than one data point for meaningful trends
    med_counts = filtered.groupby("name").size()
    trending_meds = med_counts[med_counts > 1].index.tolist()

    if trending_meds:
        # Let user pick which medicines to show on the trend chart
        default_meds = trending_meds[:5]  # show first 5 by default
        selected_meds = st.multiselect(
            "Select medicines to compare",
            options=trending_meds,
            default=default_meds,
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
            )

            # Add DRAP reference lines for selected medicines
            for med_name in selected_meds:
                drap_val = trend_data[trend_data["name"] == med_name]["drap_price_pkr"].iloc[0]
                if drap_val and drap_val > 0:
                    fig_trend.add_hline(
                        y=drap_val,
                        line_dash="dash",
                        line_color="gray",
                        opacity=0.4,
                        annotation_text=f"DRAP: {med_name}",
                        annotation_position="top left",
                        annotation_font_size=9,
                    )

            fig_trend.update_layout(
                height=450,
                margin=dict(l=10, r=10, t=40, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Select at least one medicine to see trends.")
    else:
        st.info("Not enough data points for trend analysis. Run the scraper multiple times to build history.")

    # ------ Footer ------
    st.markdown("""
    <div class="footer">
        Data sourced from <strong>Dawaai.pk</strong> |
        Compared against <strong>DRAP registered prices</strong> |
        Built by <strong>Rusham Elahi</strong>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
