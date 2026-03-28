"""
Research Insights Module — inspired by ICONIP 2024 paper
"Data-Driven Approach to Assess and Identify Gaps in Healthcare Setup in South Asia"
"""

import os
import csv
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

# Approximate average monthly income (PKR) by city — 2024 estimates
CITY_INCOME = {
    "Karachi": 65000,
    "Lahore": 55000,
    "Islamabad": 85000,
    "Rawalpindi": 50000,
    "Peshawar": 40000,
    "Faisalabad": 42000,
}

# Approximate city populations (millions) — for pharmacy density
CITY_POPULATION = {
    "Karachi": 16.0,
    "Lahore": 13.0,
    "Islamabad": 1.2,
    "Rawalpindi": 3.6,
    "Peshawar": 2.3,
    "Faisalabad": 3.4,
}

# Estimated total pharmacies per city (registered + unregistered, 2024 estimates)
# Sources: Provincial health departments, pharmacy council registrations
CITY_PHARMACIES = {
    "Karachi": 12500,
    "Lahore": 9800,
    "Islamabad": 2100,
    "Rawalpindi": 3200,
    "Peshawar": 2800,
    "Faisalabad": 3500,
}

# WHO Model List of Essential Medicines — common generics subset
WHO_ESSENTIAL_GENERICS = [
    "Paracetamol", "Ibuprofen", "Aspirin", "Amoxicillin", "Azithromycin",
    "Ciprofloxacin", "Metronidazole", "Cefixime", "Metformin", "Glimepiride",
    "Amlodipine", "Losartan", "Atenolol", "Enalapril", "Omeprazole",
    "Salbutamol", "Fluoxetine", "Diazepam", "Diclofenac",
    "Atorvastatin", "Simvastatin", "Furosemide",
    "Hydrochlorothiazide", "Warfarin", "Clopidogrel", "Dexamethasone",
    "Prednisolone", "Folic Acid", "Ferrous Sulfate", "Levothyroxine",
    "Cetirizine", "Loratadine", "Clotrimazole", "Erythromycin",
    "Levofloxacin", "Clarithromycin", "Pantoprazole", "Bisoprolol",
    "Montelukast", "Rosuvastatin", "Alprazolam", "Gabapentin",
    "Captopril", "Ranitidine", "Domperidone", "Metoclopramide",
    "Doxycycline", "Ceftriaxone", "Mefenamic Acid", "Tramadol",
]


def _load_drap_csv():
    """Load DRAP prices CSV."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", "drap_prices.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Feature 1 — Medicine Accessibility Score
# ---------------------------------------------------------------------------

def _render_accessibility_score(df, pharmacies_df, colors):
    st.markdown(
        '<div class="section-title">Medicine Accessibility Score</div>',
        unsafe_allow_html=True,
    )
    st.caption("Composite score per city based on pharmacy density, price fairness, and stock availability.")

    if pharmacies_df.empty:
        st.info("No pharmacy data available.")
        return

    cities = list(CITY_PHARMACIES.keys())
    rows = []

    for city in cities:
        pharmacy_count = CITY_PHARMACIES.get(city, 500)
        pop = CITY_POPULATION.get(city, 1.0)

        # Pharmacy density (per million people), normalized
        density_raw = pharmacy_count / pop
        # Price fairness (1 - avg overprice %, clamped)
        avg_overprice = df["overprice_pct"].mean() if not df.empty else 0
        price_fairness = max(0, min(1, 1 - avg_overprice / 100))
        # Stock availability
        if "availability" in df.columns and not df.empty:
            stock_rate = (df["availability"] == "In Stock").mean()
        else:
            stock_rate = 0.5

        rows.append({
            "City": city,
            "Pharmacies": pharmacy_count,
            "Population (M)": pop,
            "Density (per M)": round(density_raw, 2),
            "Price Fairness": round(price_fairness, 2),
            "Stock Rate": round(stock_rate, 2),
        })

    score_df = pd.DataFrame(rows)
    # Normalize density to 0-1
    max_density = score_df["Density (per M)"].max()
    if max_density > 0:
        score_df["Density Score"] = score_df["Density (per M)"] / max_density
    else:
        score_df["Density Score"] = 0

    # Composite score
    score_df["Accessibility Score"] = (
        0.4 * score_df["Density Score"]
        + 0.3 * score_df["Price Fairness"]
        + 0.3 * score_df["Stock Rate"]
    ).round(2)

    score_df = score_df.sort_values("Accessibility Score", ascending=True)

    fig = px.bar(
        score_df,
        y="City",
        x="Accessibility Score",
        orientation="h",
        color="Accessibility Score",
        color_continuous_scale=["#3f3f46", "#fafafa"],
        text="Accessibility Score",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(gridcolor=colors["border"]),
        xaxis=dict(gridcolor=colors["border"], range=[0, 1]),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        score_df[["City", "Pharmacies", "Population (M)", "Density (per M)",
                  "Price Fairness", "Stock Rate", "Accessibility Score"]],
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Feature 2 — Affordability Index
# ---------------------------------------------------------------------------

def _render_affordability_index(df, colors):
    st.markdown(
        '<div class="section-title">Affordability Index</div>',
        unsafe_allow_html=True,
    )
    st.caption("How medicine costs compare to average daily income across Pakistani cities.")

    if df.empty:
        st.info("No data available.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")

    # Essential basket: pick 5 common medicines
    basket_generics = ["Paracetamol", "Amoxicillin", "Omeprazole", "Amlodipine", "Metformin"]
    basket_items = []
    for gen in basket_generics:
        match = latest[latest["generic_name"].str.lower() == gen.lower()]
        if not match.empty:
            cheapest = match.sort_values("price_pkr").iloc[0]
            basket_items.append({"Generic": gen, "Medicine": cheapest["name"],
                                 "Price": cheapest["price_pkr"]})

    if not basket_items:
        st.info("Not enough data for affordability analysis.")
        return

    basket_df = pd.DataFrame(basket_items)
    basket_total = basket_df["Price"].sum()

    # City comparison
    city_rows = []
    for city, income in CITY_INCOME.items():
        daily_income = income / 30
        monthly_30day = basket_total * 30  # 30-day supply
        pct_income = (monthly_30day / income) * 100
        city_rows.append({
            "City": city,
            "Avg Monthly Income": f"Rs {income:,.0f}",
            "Daily Income": f"Rs {daily_income:,.0f}",
            "30-Day Basket Cost": f"Rs {monthly_30day:,.0f}",
            "% of Monthly Income": round(pct_income, 1),
            "_pct": pct_income,
        })

    city_df = pd.DataFrame(city_rows).sort_values("_pct", ascending=True)

    fig = px.bar(
        city_df,
        y="City",
        x="_pct",
        orientation="h",
        color="_pct",
        color_continuous_scale=["#3f3f46", "#ef4444"],
        text="% of Monthly Income",
        labels={"_pct": "% of Monthly Income"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(gridcolor=colors["border"]),
        xaxis=dict(gridcolor=colors["border"], title="% of Monthly Income"),
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Based on a 30-day supply of {len(basket_items)} essential medicines: "
               + ", ".join(b["Generic"] for b in basket_items))

    st.dataframe(
        city_df[["City", "Avg Monthly Income", "Daily Income", "30-Day Basket Cost", "% of Monthly Income"]],
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Feature 3 — Pharmacy Desert Map
# ---------------------------------------------------------------------------

def _render_pharmacy_desert(pharmacies_df, colors):
    st.markdown(
        '<div class="section-title">Pharmacy Coverage Map</div>',
        unsafe_allow_html=True,
    )
    st.caption("Pharmacy density relative to city population. Lower density = potential access gap.")

    if pharmacies_df.empty:
        st.info("No pharmacy data available.")
        return

    city_stats = pharmacies_df.groupby("city").agg(
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
    ).reset_index()

    city_stats["count"] = city_stats["city"].map(CITY_PHARMACIES).fillna(500).astype(int)
    city_stats["population"] = city_stats["city"].map(CITY_POPULATION).fillna(1.0)
    city_stats["density"] = (city_stats["count"] / city_stats["population"]).round(0).astype(int)
    city_stats["coverage"] = city_stats["density"].apply(
        lambda d: "Low" if d < 800 else ("Medium" if d < 1100 else "High")
    )

    color_map = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#22c55e"}

    fig = px.scatter_mapbox(
        city_stats,
        lat="lat",
        lon="lon",
        size="count",
        color="coverage",
        color_discrete_map=color_map,
        hover_name="city",
        hover_data={
            "count": True,
            "population": True,
            "density": True,
            "coverage": True,
            "lat": False,
            "lon": False,
        },
        zoom=5,
        center={"lat": 30.3753, "lon": 69.3451},
        height=450,
        size_max=30,
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(font=dict(color=colors["text"])),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        city_stats[["city", "count", "population", "density", "coverage"]].rename(columns={
            "city": "City", "count": "Pharmacies", "population": "Population (M)",
            "density": "Per Million", "coverage": "Coverage Level",
        }).sort_values("Per Million"),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Feature 4 — Stock-Out Tracker
# ---------------------------------------------------------------------------

def _render_stockout_tracker(df, colors):
    st.markdown(
        '<div class="section-title">Stock-Out Tracker</div>',
        unsafe_allow_html=True,
    )
    st.caption("Medicines most frequently reported out of stock — signals supply chain gaps.")

    if df.empty or "availability" not in df.columns:
        st.info("No availability data.")
        return

    avail = df.groupby("name")["availability"].apply(
        lambda s: (s == "Out of Stock").sum() / len(s) * 100
    ).reset_index()
    avail.columns = ["Medicine", "Stock-Out Rate (%)"]
    avail = avail.sort_values("Stock-Out Rate (%)", ascending=False)

    top = avail[avail["Stock-Out Rate (%)"] > 0].head(20)
    if top.empty:
        st.success("No stock-out events detected across tracked medicines.")
        return

    fig = px.bar(
        top,
        y="Medicine",
        x="Stock-Out Rate (%)",
        orientation="h",
        color="Stock-Out Rate (%)",
        color_continuous_scale=["#3f3f46", "#ef4444"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(300, len(top) * 28),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", gridcolor=colors["border"]),
        xaxis=dict(gridcolor=colors["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap — availability over time
    if df["scraped_at"].nunique() > 1:
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Availability Timeline</div>',
                    unsafe_allow_html=True)

        top_meds = top.head(15)["Medicine"].tolist()
        timeline = df[df["name"].isin(top_meds)].copy()
        timeline["date"] = pd.to_datetime(timeline["scraped_at"]).dt.date.astype(str)
        timeline["status_num"] = timeline["availability"].map({
            "In Stock": 1, "Limited": 0.5, "Out of Stock": 0, "Unknown": 0.25,
        }).fillna(0.25)

        # Pivot: get the dominant status label per medicine per date
        pivot_num = timeline.pivot_table(
            index="name", columns="date", values="status_num", aggfunc="mean"
        ).fillna(0.25)

        pivot_label = timeline.pivot_table(
            index="name", columns="date", values="availability", aggfunc="first"
        ).fillna("Unknown")

        # Build annotations with status text
        annotations = []
        for i, med in enumerate(pivot_num.index):
            for j, date in enumerate(pivot_num.columns):
                label = pivot_label.loc[med, date] if med in pivot_label.index and date in pivot_label.columns else ""
                short = {"In Stock": "In Stock", "Out of Stock": "Out", "Limited": "Limited", "Unknown": "?"}.get(label, "?")
                val = pivot_num.iloc[i, j]
                annotations.append(dict(
                    x=date, y=med, text=short, showarrow=False,
                    font=dict(color="#ffffff" if val < 0.3 else "#0a0a0f", size=10),
                ))

        fig_hm = go.Figure(data=go.Heatmap(
            z=pivot_num.values,
            x=pivot_num.columns.tolist(),
            y=pivot_num.index.tolist(),
            colorscale=[[0, "#ef4444"], [0.25, "#71717a"], [0.5, "#f59e0b"], [1, "#22c55e"]],
            showscale=False,
            xgap=2, ygap=2,
            hovertemplate="Medicine: %{y}<br>Date: %{x}<extra></extra>",
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(300, len(top_meds) * 32),
            xaxis=dict(gridcolor=colors["border"]),
            yaxis=dict(gridcolor=colors["border"]),
            annotations=annotations,
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Green = In Stock, Yellow = Limited, Red = Out of Stock, Grey = Unknown")


# ---------------------------------------------------------------------------
# Feature 5 — Generic Penetration Rate
# ---------------------------------------------------------------------------

def _render_generic_penetration(df, colors):
    st.markdown(
        '<div class="section-title">Generic Penetration Analysis</div>',
        unsafe_allow_html=True,
    )
    st.caption("How many medicines have a cheaper generic alternative? Shows the awareness and adoption gap.")

    if df.empty or "generic_name" not in df.columns:
        st.info("No generic data available.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")

    generics_with_multiple = latest.groupby("generic_name").filter(
        lambda g: g["name"].nunique() > 1
    )

    if generics_with_multiple.empty:
        st.info("Not enough data for generic penetration analysis.")
        return

    results = []
    for gen, group in generics_with_multiple.groupby("generic_name"):
        prices = group.groupby("name")["price_pkr"].min()
        cheapest_name = prices.idxmin()
        cheapest_price = prices.min()
        most_exp_price = prices.max()
        if most_exp_price > cheapest_price:
            saving_pct = ((most_exp_price - cheapest_price) / most_exp_price) * 100
            results.append({
                "Generic": gen,
                "Brands Available": len(prices),
                "Cheapest": cheapest_name,
                "Cheapest Price": cheapest_price,
                "Most Expensive Price": most_exp_price,
                "Potential Saving (%)": round(saving_pct, 1),
            })

    if not results:
        st.info("No price differences found between generic alternatives.")
        return

    res_df = pd.DataFrame(results).sort_values("Potential Saving (%)", ascending=False)

    # Summary metrics
    total_generics = latest["generic_name"].nunique()
    with_alternatives = len(res_df)
    avg_saving = res_df["Potential Saving (%)"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Generic Classes", total_generics)
    col2.metric("With Cheaper Alternative", with_alternatives)
    col3.metric("Avg Potential Saving", f"{avg_saving:.0f}%")

    fig = px.bar(
        res_df.head(15),
        y="Generic",
        x="Potential Saving (%)",
        orientation="h",
        color="Potential Saving (%)",
        color_continuous_scale=["#3f3f46", "#22c55e"],
        hover_data=["Cheapest", "Cheapest Price", "Most Expensive Price"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(300, min(len(res_df.head(15)), 15) * 28),
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed", gridcolor=colors["border"]),
        xaxis=dict(gridcolor=colors["border"]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Feature 6 — Essential Medicine Coverage
# ---------------------------------------------------------------------------

def _render_essential_coverage(df, colors):
    st.markdown(
        '<div class="section-title">WHO Essential Medicine Coverage</div>',
        unsafe_allow_html=True,
    )
    st.caption("Cross-referencing tracked medicines against the WHO Model List of Essential Medicines.")

    if df.empty or "generic_name" not in df.columns:
        st.info("No data available.")
        return

    tracked_generics = set(
        g.strip().lower() for g in df["generic_name"].dropna().unique()
    )

    covered = []
    missing = []
    for gen in WHO_ESSENTIAL_GENERICS:
        if gen.lower() in tracked_generics:
            # Find cheapest
            match = df[df["generic_name"].str.lower() == gen.lower()]
            latest = match.sort_values("scraped_at").drop_duplicates(subset=["name"], keep="last")
            cheapest = latest.sort_values("price_pkr").iloc[0]
            in_stock = (latest["availability"] == "In Stock").any()
            covered.append({
                "Generic": gen,
                "Available As": cheapest["name"],
                "Price (PKR)": cheapest["price_pkr"],
                "In Stock": "Yes" if in_stock else "No",
                "Status": "Covered",
            })
        else:
            missing.append({"Generic": gen, "Status": "Gap"})

    coverage_pct = len(covered) / len(WHO_ESSENTIAL_GENERICS) * 100

    # Donut chart
    fig = go.Figure(data=[go.Pie(
        labels=["Covered", "Missing"],
        values=[len(covered), len(missing)],
        hole=0.6,
        marker=dict(colors=["#22c55e", "#ef4444"]),
        textfont=dict(color="#ffffff", size=13),
    )])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                    font=dict(color=colors["text"])),
        annotations=[dict(
            text=f"{coverage_pct:.0f}%",
            x=0.5, y=0.5, font_size=28, font_color="#ffffff",
            showarrow=False,
        )],
    )
    st.plotly_chart(fig, use_container_width=True)

    if covered:
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Covered Essential Medicines</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(covered)[["Generic", "Available As", "Price (PKR)", "In Stock"]],
                     hide_index=True, use_container_width=True)

    if missing:
        st.markdown('<div class="section-title" style="font-size:0.9rem;">Coverage Gaps</div>',
                    unsafe_allow_html=True)
        st.caption("These WHO essential medicines are not currently tracked in the database.")
        gap_cols = st.columns(4)
        for i, m in enumerate(missing):
            gap_cols[i % 4].markdown(
                f'<span class="pill pill-gray" style="margin:2px;">{m["Generic"]}</span>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Feature 7 — Regional Price Disparity
# ---------------------------------------------------------------------------

def _render_price_disparity(df, colors):
    st.markdown(
        '<div class="section-title">Price Disparity Across Sources</div>',
        unsafe_allow_html=True,
    )
    st.caption("Comparing medicine prices between pharmacy sources to identify pricing gaps.")

    if df.empty or df["source"].nunique() < 2:
        st.info("Need at least 2 sources for price comparison.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")
    pivot = latest.pivot_table(index="name", columns="source", values="price_pkr", aggfunc="first")
    pivot = pivot.dropna()

    if pivot.empty or len(pivot) < 2:
        st.info("Not enough overlapping medicines across sources.")
        return

    sources = pivot.columns.tolist()
    pivot["disparity"] = abs(pivot[sources[0]] - pivot[sources[1]])
    pivot["disparity_pct"] = (pivot["disparity"] / pivot[[sources[0], sources[1]]].mean(axis=1) * 100).round(1)
    pivot = pivot.sort_values("disparity_pct", ascending=False)

    top = pivot.head(20).reset_index()

    fig = go.Figure()
    for src in sources:
        fig.add_trace(go.Bar(
            name=src,
            y=top["name"],
            x=top[src],
            orientation="h",
        ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(350, len(top) * 28),
        yaxis=dict(autorange="reversed", gridcolor=colors["border"]),
        xaxis=dict(gridcolor=colors["border"], title="Price (PKR)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(color=colors["text"])),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    avg_disparity = pivot["disparity_pct"].mean()
    max_disparity_med = pivot.index[0]
    max_disparity_val = pivot["disparity_pct"].iloc[0]
    st.caption(
        f"Average price disparity: **{avg_disparity:.1f}%** | "
        f"Largest gap: **{max_disparity_med}** ({max_disparity_val:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Feature 8 — Anomaly Detection Dashboard
# ---------------------------------------------------------------------------

def _render_anomaly_dashboard(df, colors):
    st.markdown(
        '<div class="section-title">Price Anomaly Detection</div>',
        unsafe_allow_html=True,
    )
    st.caption("Using Isolation Forest algorithm to flag unusual pricing patterns — potential gouging or data errors.")

    if df.empty:
        st.info("No data available.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name"], keep="last").copy()
    if len(latest) < 5:
        st.info("Not enough data for anomaly detection.")
        return

    features = latest[["price_pkr", "overprice_pct"]].fillna(0).values
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(features)
    latest["anomaly"] = (preds == -1).astype(int)

    n_anomalies = latest["anomaly"].sum()
    n_total = len(latest)

    col1, col2 = st.columns(2)
    col1.metric("Total Medicines Analyzed", n_total)
    col2.metric("Anomalies Detected", n_anomalies)

    # Scatter plot
    latest["Status"] = latest["anomaly"].map({0: "Normal", 1: "Anomaly"})
    fig = px.scatter(
        latest,
        x="price_pkr",
        y="overprice_pct",
        color="Status",
        color_discrete_map={"Normal": "#3f3f46", "Anomaly": "#ef4444"},
        hover_name="name",
        hover_data=["source", "price_pkr", "drap_price_pkr", "overprice_pct"],
        labels={"price_pkr": "Price (PKR)", "overprice_pct": "Overprice %"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        xaxis=dict(gridcolor=colors["border"]),
        yaxis=dict(gridcolor=colors["border"]),
        legend=dict(font=dict(color=colors["text"])),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    anomalies = latest[latest["anomaly"] == 1][
        ["name", "brand", "price_pkr", "drap_price_pkr", "overprice_pct", "source"]
    ].sort_values("overprice_pct", ascending=False)
    anomalies.columns = ["Medicine", "Brand", "Price", "DRAP Price", "Overprice %", "Source"]
    if not anomalies.empty:
        st.dataframe(anomalies, hide_index=True, use_container_width=True)

    st.caption(
        "Isolation Forest is an unsupervised ML algorithm that identifies data points "
        "deviating significantly from the norm — used in our ICONIP 2024 research "
        "to detect healthcare pricing anomalies across South Asia."
    )


# ---------------------------------------------------------------------------
# Feature 9 — Trend Forecasting
# ---------------------------------------------------------------------------

def _render_trend_forecast(df, colors):
    st.markdown(
        '<div class="section-title">Price Trend Forecasting</div>',
        unsafe_allow_html=True,
    )
    st.caption("Simple linear projection based on historical price data. Indicative only.")

    if df.empty:
        st.info("No data available.")
        return

    # Only medicines with multiple data points
    med_counts = df.groupby("name")["scraped_at"].nunique()
    trending = med_counts[med_counts > 1].index.tolist()

    if not trending:
        st.info("Need multiple scrape dates for trend analysis.")
        return

    selected = st.selectbox("Select medicine", options=sorted(trending), key="forecast_med")

    if not selected:
        return

    med_data = df[df["name"] == selected].copy()
    med_data["scraped_at"] = pd.to_datetime(med_data["scraped_at"])
    med_data = med_data.sort_values("scraped_at").drop_duplicates(subset=["scraped_at"], keep="last")

    if len(med_data) < 2:
        st.info("Not enough data points for this medicine.")
        return

    # Convert dates to numeric (days from first date)
    t0 = med_data["scraped_at"].iloc[0]
    med_data["days"] = (med_data["scraped_at"] - t0).dt.total_seconds() / 86400
    x = med_data["days"].values
    y = med_data["price_pkr"].values

    # Linear fit
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    # Forecast 28 days ahead
    last_day = x[-1]
    forecast_days = np.array([last_day + 7 * i for i in range(1, 5)])
    forecast_prices = np.polyval(coeffs, forecast_days)
    forecast_dates = [t0 + pd.Timedelta(days=d) for d in forecast_days]

    # Residual std for confidence band
    fitted = np.polyval(coeffs, x)
    residual_std = np.std(y - fitted) if len(y) > 2 else y.std() * 0.1

    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=med_data["scraped_at"],
        y=med_data["price_pkr"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#fafafa", width=2),
        marker=dict(size=8),
    ))

    # Trend line
    all_days = np.concatenate([x, forecast_days])
    all_dates = list(med_data["scraped_at"]) + forecast_dates
    trend_line = np.polyval(coeffs, all_days)
    fig.add_trace(go.Scatter(
        x=all_dates,
        y=trend_line,
        mode="lines",
        name="Trend",
        line=dict(color="#a1a1aa", width=1, dash="dash"),
    ))

    # Forecast points
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode="markers",
        name="Forecast",
        marker=dict(size=10, color="#f59e0b", symbol="diamond"),
    ))

    # Confidence band
    upper = forecast_prices + 1.96 * residual_std
    lower = np.maximum(0, forecast_prices - 1.96 * residual_std)
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(245,158,11,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence",
    ))

    # DRAP reference
    drap = med_data["drap_price_pkr"].iloc[0]
    if drap and drap > 0:
        fig.add_hline(
            y=drap, line_dash="dot", line_color="#71717a",
            annotation_text=f"DRAP: Rs {drap:.0f}",
            annotation_font_size=10, annotation_font_color="#71717a",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        xaxis=dict(gridcolor=colors["border"], title="Date"),
        yaxis=dict(gridcolor=colors["border"], title="Price (PKR)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                    font=dict(color=colors["text"])),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    trend_dir = "increasing" if slope > 0.1 else ("decreasing" if slope < -0.1 else "stable")
    st.caption(
        f"Trend: **{trend_dir}** (Rs {slope:+.2f}/day) | "
        f"4-week forecast range: Rs {forecast_prices[-1] - 1.96*residual_std:.0f} — "
        f"Rs {forecast_prices[-1] + 1.96*residual_std:.0f}"
    )


# ---------------------------------------------------------------------------
# Feature 10 — Research Attribution
# ---------------------------------------------------------------------------

def _render_research_attribution(colors):
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 2rem;
    ">
        <div style="font-size:0.75rem; font-weight:600; text-transform:uppercase;
                    letter-spacing:0.06em; color:{colors['text_dim']}; margin-bottom:0.6rem;">
            Research Foundation
        </div>
        <div style="font-size:1.1rem; font-weight:700; color:#ffffff; margin-bottom:0.5rem;">
            Data-Driven Approach to Assess and Identify Gaps in Healthcare Setup in South Asia
        </div>
        <div style="color:{colors['text']}; font-size:0.85rem; line-height:1.6;">
            Published at <strong>ICONIP 2024</strong> (International Conference on Neural Information Processing)
            <br><br>
            This dashboard implements and extends the research methodology — using web-scraped pharmaceutical data,
            anomaly detection (Isolation Forest), and accessibility scoring to identify healthcare gaps
            in Pakistan's medicine supply chain.
            <br><br>
            The features in this tab directly reflect the paper's framework: quantifying access disparities,
            detecting pricing anomalies, measuring essential medicine coverage, and forecasting supply trends.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Feature 11 — Medicine Price Heatmap
# ---------------------------------------------------------------------------

def _render_price_heatmap(df, colors):
    st.markdown(
        '<div class="section-title">Medicine Price Heatmap</div>',
        unsafe_allow_html=True,
    )
    st.caption("Price intensity across medicines and pharmacy sources. Darker = more expensive.")

    if df.empty or df["source"].nunique() < 2:
        st.info("Need multiple sources for heatmap.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")
    pivot = latest.pivot_table(index="name", columns="source", values="price_pkr", aggfunc="first")
    pivot = pivot.dropna()

    if len(pivot) < 2:
        st.info("Not enough data for price heatmap.")
        return

    # Sort by average price descending, take top 30
    pivot["_avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_avg", ascending=False).head(30).drop(columns="_avg")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#0a0a0f"], [0.3, "#3f3f46"], [0.6, "#f59e0b"], [1, "#ef4444"]],
        hovertemplate="Medicine: %{y}<br>Source: %{x}<br>Price: Rs %{z:,.0f}<extra></extra>",
        colorbar=dict(title=dict(text="PKR", font=dict(color=colors["text"])), tickfont=dict(color=colors["text"])),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(400, len(pivot) * 22),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Feature 12 — City Affordability Heatmap
# ---------------------------------------------------------------------------

def _render_affordability_heatmap(df, colors):
    st.markdown(
        '<div class="section-title">City Affordability Heatmap</div>',
        unsafe_allow_html=True,
    )
    st.caption("Cost of essential medicines as percentage of daily income across cities. Red = less affordable.")

    if df.empty or "generic_name" not in df.columns:
        st.info("No data available.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")

    # Pick essential generics that exist in data
    essentials = []
    for gen in ["Paracetamol", "Amoxicillin", "Omeprazole", "Amlodipine", "Metformin",
                "Ciprofloxacin", "Azithromycin", "Losartan", "Atorvastatin", "Cetirizine",
                "Ibuprofen", "Diclofenac", "Pantoprazole", "Cefixime", "Montelukast"]:
        match = latest[latest["generic_name"].str.lower() == gen.lower()]
        if not match.empty:
            cheapest = match.sort_values("price_pkr").iloc[0]
            essentials.append({"generic": gen, "price": cheapest["price_pkr"]})

    if len(essentials) < 3:
        st.info("Not enough essential medicines in database for heatmap.")
        return

    cities = list(CITY_INCOME.keys())
    z_data = []
    y_labels = []
    for med in essentials:
        row = []
        for city in cities:
            daily_income = CITY_INCOME[city] / 30
            pct = (med["price"] / daily_income) * 100
            row.append(round(pct, 1))
        z_data.append(row)
        y_labels.append(med["generic"])

    # Build annotation text
    annotations = []
    for i, med in enumerate(essentials):
        for j, city in enumerate(cities):
            annotations.append(dict(
                x=city, y=med["generic"],
                text=f"{z_data[i][j]:.1f}%",
                showarrow=False,
                font=dict(color="#ffffff" if z_data[i][j] > 8 else "#d4d4d8", size=11),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=cities,
        y=y_labels,
        colorscale=[[0, "#134e4a"], [0.35, "#3f3f46"], [0.65, "#f59e0b"], [1, "#ef4444"]],
        hovertemplate="Medicine: %{y}<br>City: %{x}<br>% of Daily Income: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="% Income", font=dict(color=colors["text"])),
                      tickfont=dict(color=colors["text"])),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(350, len(essentials) * 32),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
        annotations=annotations,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Medicine prices are uniform (online pharmacies), but affordability varies by city income. "
        "Lower-income cities like Peshawar and Faisalabad face a higher burden for the same medicines."
    )


# ---------------------------------------------------------------------------
# Feature 13 — Overpricing Heatmap
# ---------------------------------------------------------------------------

def _render_overpricing_heatmap(df, colors):
    st.markdown(
        '<div class="section-title">Overpricing Heatmap</div>',
        unsafe_allow_html=True,
    )
    st.caption("How much each medicine exceeds its DRAP-regulated price, by source. Red = heavily overpriced.")

    if df.empty:
        st.info("No data available.")
        return

    latest = df.sort_values("scraped_at").drop_duplicates(subset=["name", "source"], keep="last")
    pivot = latest.pivot_table(index="name", columns="source", values="overprice_pct", aggfunc="first")
    pivot = pivot.dropna()

    if len(pivot) < 2:
        st.info("Not enough data for overpricing heatmap.")
        return

    # Only show overpriced medicines, sorted by max overprice
    pivot["_max"] = pivot.max(axis=1)
    pivot = pivot[pivot["_max"] > 0].sort_values("_max", ascending=False).head(25).drop(columns="_max")

    if pivot.empty:
        st.success("No overpriced medicines detected.")
        return

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#22c55e"], [0.2, "#3f3f46"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        hovertemplate="Medicine: %{y}<br>Source: %{x}<br>Overprice: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="% Above DRAP", font=dict(color=colors["text"])),
                      tickfont=dict(color=colors["text"])),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(400, len(pivot) * 22),
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_research_insights(df, pharmacies_df, colors, plotly_layout):
    """Render the full Research Insights tab with 4 sub-tabs."""

    sub_overview, sub_access, sub_price, sub_trends = st.tabs([
        "Overview & Scores",
        "Access & Coverage",
        "Price Intelligence",
        "Trends & Forecasting",
    ])

    _divider = '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.08);margin:1.5rem 0;">'

    with sub_overview:
        _render_accessibility_score(df, pharmacies_df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_affordability_index(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_affordability_heatmap(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_research_attribution(colors)

    with sub_access:
        _render_pharmacy_desert(pharmacies_df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_generic_penetration(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_essential_coverage(df, colors)

    with sub_price:
        _render_price_heatmap(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_overpricing_heatmap(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_price_disparity(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_anomaly_dashboard(df, colors)

    with sub_trends:
        _render_stockout_tracker(df, colors)
        st.markdown(_divider, unsafe_allow_html=True)
        _render_trend_forecast(df, colors)
