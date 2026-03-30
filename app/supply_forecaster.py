"""
Supply Forecaster
=================
Converts ML model predictions into human-readable shortage timelines.
"This medicine will be short in X weeks."

Uses:
- Random Forest stock-out classifier (risk_score 0-1)
- Gradient Boosting price predictor (price change %)
- Availability history degradation patterns
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from app.ml_models import (
    train_stockout_model, predict_stockout,
    train_price_model, predict_prices,
    evaluate_stockout_model,
)


# ---------------------------------------------------------------------------
# Core: Convert risk score → timeline
# ---------------------------------------------------------------------------

def estimate_shortage_timeline(risk_score, price_change_pct, avail_history):
    """Convert stock-out probability + price signals into a timeline estimate.

    Args:
        risk_score: float 0-1 from Random Forest
        price_change_pct: float from Gradient Boosting (predicted price change %)
        avail_history: list of availability strings, most recent last

    Returns:
        dict with weeks_estimate, label, severity, factors
    """
    price_rising = price_change_pct > 1
    price_spiking = price_change_pct > 10

    # Check for availability degradation pattern
    degrading = False
    if len(avail_history) >= 3:
        avail_map = {"In Stock": 0, "Limited": 1, "Out of Stock": 2, "Unknown": 1}
        recent = [avail_map.get(a, 1) for a in avail_history[-3:]]
        degrading = recent[-1] > recent[0]  # getting worse

    # Base timeline from risk score
    if risk_score >= 0.8:
        weeks = 2
        label = "within 1-2 weeks"
        severity = "critical"
    elif risk_score >= 0.6 and price_rising:
        weeks = 3
        label = "within 2-4 weeks"
        severity = "high"
    elif risk_score >= 0.6:
        weeks = 5
        label = "within 4-6 weeks"
        severity = "high"
    elif risk_score >= 0.3:
        weeks = 7
        label = "within 6-8 weeks"
        severity = "medium"
    else:
        weeks = 0
        label = "supply looks stable"
        severity = "low"

    # Accelerate by 1 tier if degrading or spiking
    if severity != "low" and (degrading or price_spiking):
        weeks = max(1, weeks - 2)
        if weeks <= 2:
            label = "within 1-2 weeks"
            severity = "critical"
        elif weeks <= 4:
            label = "within 2-4 weeks"
            severity = "high"

    # Build contributing factors
    factors = []
    if risk_score >= 0.6:
        factors.append(f"High stock-out probability ({risk_score:.0%})")
    elif risk_score >= 0.3:
        factors.append(f"Elevated stock-out probability ({risk_score:.0%})")

    if price_spiking:
        factors.append(f"Price spiking ({price_change_pct:+.1f}%) — supply pressure signal")
    elif price_rising:
        factors.append(f"Price trending up ({price_change_pct:+.1f}%)")

    if degrading:
        factors.append("Availability degrading across recent scrapes")

    oos_count = avail_history.count("Out of Stock") if avail_history else 0
    if oos_count > 0:
        factors.append(f"Already out of stock in {oos_count} recent record(s)")

    if not factors:
        factors.append("No risk indicators detected")

    return {
        "weeks": weeks,
        "label": label,
        "severity": severity,
        "factors": factors,
        "risk_score": risk_score,
        "price_change_pct": price_change_pct,
    }


def compute_all_alerts(df):
    """Batch-predict shortage timelines for all medicines."""
    so_model, so_le, so_metrics = train_stockout_model(df)
    p_model, p_le, p_metrics = train_price_model(df)

    if so_model is None:
        return [], None, None

    price_preds = predict_prices(p_model, p_le, df) if p_model else pd.DataFrame()

    alerts = []
    for med_name in sorted(df["name"].unique()):
        # Stock-out prediction
        pred = predict_stockout(so_model, so_le, df, med_name)

        # Price change prediction
        price_change = 0
        if not price_preds.empty:
            med_price = price_preds[price_preds["Medicine"] == med_name]
            if not med_price.empty:
                price_change = med_price["Change (%)"].iloc[0]

        # Availability history
        med_df = df[df["name"] == med_name].sort_values("scraped_at")
        avail_history = med_df["availability"].tolist()[-10:]

        timeline = estimate_shortage_timeline(
            pred["risk_score"], price_change, avail_history
        )
        timeline["medicine"] = med_name
        timeline["generic"] = med_df.iloc[-1].get("generic_name", "") if not med_df.empty else ""
        timeline["current_price"] = med_df.iloc[-1]["price_pkr"] if not med_df.empty else 0
        timeline["source"] = med_df.iloc[-1].get("source", "") if not med_df.empty else ""

        alerts.append(timeline)

    return alerts, so_metrics, p_metrics


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "critical": {"color": "#ef4444", "bg": "rgba(239,68,68,0.1)", "border": "rgba(239,68,68,0.3)"},
    "high":     {"color": "#f59e0b", "bg": "rgba(245,158,11,0.1)", "border": "rgba(245,158,11,0.3)"},
    "medium":   {"color": "#fb923c", "bg": "rgba(251,146,60,0.08)", "border": "rgba(251,146,60,0.2)"},
    "low":      {"color": "#22c55e", "bg": "rgba(34,197,94,0.08)", "border": "rgba(34,197,94,0.2)"},
}


def render_supply_forecaster_tab(df, colors, plotly_layout):
    """Render the Supply Forecaster tab."""

    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(239,68,68,0.06) 0%, rgba(245,158,11,0.04) 100%);
                backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.06);
                border-radius:16px;padding:1.5rem 1.75rem;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px rgba(239,68,68,0.4);"></div>
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#71717a;">Machine Learning</div>
        </div>
        <div style="font-size:1.2rem;font-weight:700;color:#fff;">Supply Shortage Forecaster</div>
        <div style="font-size:0.8rem;color:#a1a1aa;margin-top:6px;line-height:1.5;">
            Random Forest + Gradient Boosting models predict which medicines will go short and when,
            trained on real price and availability signals scraped from Pakistani pharmacies.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training models and generating predictions..."):
        alerts, so_metrics, p_metrics = compute_all_alerts(df)

    if not alerts:
        st.warning("Not enough historical data to generate predictions. Run the scraper a few more times to build history.")
        return

    # --- Model Pipeline Banner ---
    eval_metrics = evaluate_stockout_model(df)
    roc_auc = eval_metrics["roc_auc"] if eval_metrics else 0
    f1 = eval_metrics["f1"] if eval_metrics else 0
    train_n = so_metrics["train_size"] if so_metrics else 0
    price_r2 = p_metrics["cv_r2_mean"] if p_metrics else 0

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:12px;padding:0.85rem 1.25rem;margin-bottom:1.25rem;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.75rem;">
            <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:#52525b;">Stock-Out Model</div>
                    <div style="font-size:0.8rem;color:#e4e4e7;">Random Forest <span style="color:#71717a;">· 100 trees, depth 6</span></div>
                </div>
                <div>
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:#52525b;">Price Model</div>
                    <div style="font-size:0.8rem;color:#e4e4e7;">Gradient Boosting <span style="color:#71717a;">· 100 trees, lr 0.1</span></div>
                </div>
                <div>
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:#52525b;">Training Data</div>
                    <div style="font-size:0.8rem;color:#e4e4e7;">{train_n:,} samples <span style="color:#71717a;">· 8 features</span></div>
                </div>
            </div>
            <div style="display:flex;gap:1.25rem;">
                <div style="text-align:center;">
                    <div style="font-size:1.1rem;font-weight:700;color:#22c55e;">{roc_auc:.3f}</div>
                    <div style="font-size:0.6rem;color:#52525b;">ROC AUC</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem;font-weight:700;color:#f59e0b;">{f1:.2f}</div>
                    <div style="font-size:0.6rem;color:#52525b;">F1 Score</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:1.1rem;font-weight:700;color:#a1a1aa;">{price_r2:.3f}</div>
                    <div style="font-size:0.6rem;color:#52525b;">Price R²</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Categorize alerts
    critical = [a for a in alerts if a["severity"] == "critical"]
    high = [a for a in alerts if a["severity"] == "high"]
    medium = [a for a in alerts if a["severity"] == "medium"]
    low = [a for a in alerts if a["severity"] == "low"]

    # --- Summary cards ---
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, items, sev_name, sev_label in [
        (sc1, critical, "critical", "Critical"),
        (sc2, high, "high", "High Risk"),
        (sc3, medium, "medium", "Medium"),
        (sc4, low, "low", "Stable"),
    ]:
        sc = SEVERITY_COLORS[sev_name]
        with col:
            st.markdown(f"""
            <div style="background:{sc['bg']};border:1px solid {sc['border']};border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-size:2rem;font-weight:800;color:{sc['color']};">{len(items)}</div>
                <div style="font-size:0.75rem;color:{sc['color']};font-weight:600;">{sev_label}</div>
            </div>""", unsafe_allow_html=True)

    # --- Alert cards ---
    if critical or high:
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Shortage Alerts</div>', unsafe_allow_html=True)

        for alert in critical + high:
            sc = SEVERITY_COLORS[alert["severity"]]
            factors_html = "".join(
                f'<div style="font-size:0.75rem;color:{colors["text"]};margin:2px 0;">- {f}</div>'
                for f in alert["factors"]
            )
            sev_label = "CRITICAL" if alert["severity"] == "critical" else "HIGH RISK"

            st.markdown(f"""
            <div style="background:{sc['bg']};border:1px solid {sc['border']};border-left:4px solid {sc['color']};
                        border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:0.5rem;">
                    <div style="flex:1;">
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                            <span style="color:#fff;font-weight:700;font-size:1rem;">{alert['medicine']}</span>
                            <span style="background:{sc['color']};color:#fff;font-size:0.6rem;padding:2px 8px;border-radius:4px;
                                         font-weight:700;text-transform:uppercase;">{sev_label}</span>
                        </div>
                        <div style="color:#a1a1aa;font-size:0.8rem;">{alert['generic']}</div>
                        <div style="display:flex;gap:6px;margin:6px 0;">
                            <span style="font-size:0.6rem;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,0.06);
                                         color:#71717a;letter-spacing:0.03em;">RF classifier</span>
                            <span style="font-size:0.6rem;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,0.06);
                                         color:#71717a;letter-spacing:0.03em;">GB regressor</span>
                        </div>
                        {factors_html}
                    </div>
                    <div style="text-align:right;">
                        <div style="color:{sc['color']};font-size:1.3rem;font-weight:800;">Shortage {alert['label']}</div>
                        <div style="color:#71717a;font-size:0.75rem;">Risk score: {alert['risk_score']:.0%}</div>
                        <div style="color:#a1a1aa;font-size:0.8rem;">Rs {alert['current_price']:,.0f} at {alert['source']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if medium:
        st.markdown('<div class="section-title" style="margin-top:1rem;">Watch List</div>', unsafe_allow_html=True)
        cards_html = ""
        for alert in medium:
            sc = SEVERITY_COLORS["medium"]
            cards_html += f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid {sc['border']};border-left:3px solid {sc['color']};
                        border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.4rem;display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="color:#e4e4e7;font-weight:600;font-size:0.85rem;">{alert['medicine']}</span>
                    <span style="color:#71717a;font-size:0.75rem;margin-left:8px;">{alert['generic']}</span>
                </div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="color:{sc['color']};font-size:0.8rem;">{alert['label']}</span>
                    <span style="color:{sc['color']};font-weight:700;font-size:0.85rem;">{alert['risk_score']:.0%}</span>
                </div>
            </div>"""
        st.markdown(cards_html, unsafe_allow_html=True)

    with st.expander(f"Stable Medicines ({len(low)})"):
        if low:
            stable_df = pd.DataFrame([{
                "Medicine": a["medicine"],
                "Generic": a["generic"],
                "Risk Score": f"{a['risk_score']:.0%}",
                "Price": f"Rs {a['current_price']:,.0f}",
            } for a in low])
            st.dataframe(stable_df, hide_index=True, use_container_width=True)

    # --- Individual Medicine Deep Dive ---
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Medicine Deep Dive</div>', unsafe_allow_html=True)

    all_meds = sorted(df["name"].unique())
    selected = st.selectbox("Select a medicine", all_meds, key="forecast_medicine_select")

    if selected:
        # Find this medicine's alert
        med_alert = next((a for a in alerts if a["medicine"] == selected), None)
        if med_alert:
            sc = SEVERITY_COLORS[med_alert["severity"]]

            # Risk gauge + timeline
            col_gauge, col_timeline = st.columns([1, 2])

            with col_gauge:
                # Risk score gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=med_alert["risk_score"] * 100,
                    number={"suffix": "%", "font": {"color": "#fff", "size": 32}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#71717a"}},
                        "bar": {"color": sc["color"]},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "bordercolor": "rgba(255,255,255,0.08)",
                        "steps": [
                            {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                            {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                            {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                        ],
                    },
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#e4e4e7"),
                    margin=dict(l=20, r=20, t=30, b=10), height=200,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_timeline:
                # Timeline bar
                weeks_markers = [1, 2, 4, 6, 8]
                predicted_week = med_alert["weeks"] if med_alert["weeks"] > 0 else None

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px;padding:1rem 1.25rem;">
                    <div style="font-size:0.7rem;text-transform:uppercase;color:#71717a;margin-bottom:8px;">Shortage Timeline</div>
                    <div style="font-size:1.1rem;font-weight:700;color:{sc['color']};">
                        {med_alert['label'].title() if med_alert['severity'] != 'low' else 'No shortage expected'}
                    </div>
                    <div style="display:flex;align-items:center;margin-top:12px;gap:0px;">
                        {''.join(f'''<div style="flex:1;text-align:center;">
                            <div style="height:8px;background:{'linear-gradient(90deg, ' + sc['color'] + ', ' + sc['color'] + ')' if predicted_week and w <= predicted_week else 'rgba(255,255,255,0.08)'};
                                        border-radius:{'4px 0 0 4px' if w == 1 else ('0 4px 4px 0' if w == 8 else '0')};"></div>
                            <div style="font-size:0.7rem;color:#71717a;margin-top:4px;">{w}w</div>
                        </div>''' for w in weeks_markers)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Contributing factors
            if med_alert["factors"]:
                factors_html = "".join(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
                    f'<div style="width:6px;height:6px;border-radius:50%;background:{sc["color"]};flex-shrink:0;"></div>'
                    f'<span style="color:{colors["text"]};font-size:0.85rem;">{f}</span></div>'
                    for f in med_alert["factors"]
                )
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px;padding:1rem 1.25rem;margin-top:0.75rem;">
                    <div style="font-size:0.7rem;text-transform:uppercase;color:#71717a;margin-bottom:8px;">Contributing Factors</div>
                    {factors_html}
                </div>
                """, unsafe_allow_html=True)

        # Price trajectory chart (only if there's an actual trend to show)
        med_df = df[df["name"] == selected].sort_values("scraped_at").copy()
        med_df["scraped_at"] = pd.to_datetime(med_df["scraped_at"])
        if len(med_df["scraped_at"].dt.date.unique()) > 1:
            drap_price = med_df.iloc[-1].get("drap_price_pkr", 0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=med_df["scraped_at"], y=med_df["price_pkr"],
                mode="lines+markers", name="Price",
                line=dict(color="#fafafa", width=2), marker=dict(size=5),
            ))
            if drap_price > 0:
                fig.add_hline(y=drap_price, line_dash="dot", line_color="#71717a",
                              annotation_text=f"DRAP: Rs {drap_price:.0f}",
                              annotation_font_size=10, annotation_font_color="#71717a")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                margin=dict(l=10, r=10, t=30, b=10), height=280,
                title="Price History",
                xaxis=dict(gridcolor=colors["border"]),
                yaxis=dict(gridcolor=colors["border"], title="Price (PKR)"),
                hoverlabel=dict(bgcolor="#1a1a1f", bordercolor="#3f3f46", font_color="#e4e4e7"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Availability heatmap (source × time)
        if not med_df.empty and len(med_df) > 1:
            avail_map = {"In Stock": 1, "Limited": 0.5, "Out of Stock": 0, "Unknown": 0.25}
            med_df["avail_num"] = med_df["availability"].map(avail_map).fillna(0.25)
            med_df["date"] = med_df["scraped_at"].dt.strftime("%b %d")

            pivot = med_df.pivot_table(
                index="source", columns="date", values="avail_num", aggfunc="last"
            )
            if not pivot.empty and pivot.shape[1] > 1:
                fig_hm = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=[[0, "#ef4444"], [0.25, "#f59e0b"], [0.5, "#f59e0b"], [1, "#22c55e"]],
                    showscale=False,
                    hovertemplate="Source: %{y}<br>Date: %{x}<br>Stock: %{z:.0%}<extra></extra>",
                    xgap=2, ygap=2,
                ))
                # Add text annotations
                annotations = []
                labels_map = {1: "In Stock", 0.5: "Limited", 0: "OOS", 0.25: "?"}
                for i, source in enumerate(pivot.index):
                    for j, date in enumerate(pivot.columns):
                        val = pivot.values[i][j]
                        if not np.isnan(val):
                            annotations.append(dict(
                                x=date, y=source,
                                text=labels_map.get(val, "?"),
                                showarrow=False,
                                font=dict(color="#fff" if val < 0.5 else "#000", size=9),
                            ))
                fig_hm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                    margin=dict(l=10, r=10, t=30, b=10),
                    title="Availability History",
                    height=max(150, len(pivot.index) * 45 + 60),
                    xaxis=dict(side="top"),
                    yaxis=dict(autorange="reversed"),
                    annotations=annotations,
                )
                st.plotly_chart(fig_hm, use_container_width=True)

    # --- Model Evaluation ---
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Model Evaluation & Feature Analysis</div>', unsafe_allow_html=True)
    with st.expander("ROC Curve, Confusion Matrix & Feature Importance", expanded=True):
        eval_metrics = evaluate_stockout_model(df)
        if eval_metrics:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Precision", f"{eval_metrics['precision']:.2f}")
            mc2.metric("Recall", f"{eval_metrics['recall']:.2f}")
            mc3.metric("F1 Score", f"{eval_metrics['f1']:.2f}")
            mc4.metric("ROC AUC", f"{eval_metrics['roc_auc']:.3f}")

            roc_col, cm_col = st.columns(2)

            with roc_col:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=eval_metrics["fpr"], y=eval_metrics["tpr"],
                    mode="lines", name=f"RF (AUC={eval_metrics['roc_auc']:.3f})",
                    line=dict(color="#22c55e", width=2),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random",
                    line=dict(color="#3f3f46", width=1, dash="dash"),
                ))
                fig_roc.update_layout(
                    title="ROC Curve",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                    margin=dict(l=10, r=10, t=35, b=10), height=300,
                    xaxis=dict(title="False Positive Rate", gridcolor=colors["border"]),
                    yaxis=dict(title="True Positive Rate", gridcolor=colors["border"]),
                    legend=dict(font=dict(color=colors["text"]), x=0.4, y=0.1),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with cm_col:
                cm = eval_metrics["confusion_matrix"]
                labels = ["In Stock", "Out of Stock"]
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels,
                    colorscale=[[0, "#0a0a0f"], [0.5, "#3f3f46"], [1, "#22c55e"]],
                    showscale=False,
                ))
                annotations = []
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        annotations.append(dict(
                            x=labels[j], y=labels[i], text=str(cm[i][j]),
                            showarrow=False, font=dict(color="#fff", size=16),
                        ))
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                    margin=dict(l=10, r=10, t=35, b=10), height=300,
                    xaxis=dict(title="Predicted", side="bottom"),
                    yaxis=dict(title="Actual", autorange="reversed"),
                    annotations=annotations,
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance
            imp = eval_metrics["feature_importances"]
            imp_df = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
            imp_df = imp_df.sort_values("Importance", ascending=True)
            imp_df["Contribution"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(1)

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=imp_df["Feature"], x=imp_df["Contribution"],
                orientation="h",
                marker_color=imp_df["Importance"].apply(
                    lambda v: "#22c55e" if v > imp_df["Importance"].median() else "#3f3f46"
                ).tolist(),
                text=imp_df["Contribution"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                textfont=dict(color=colors["text"], size=11),
            ))
            fig_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                margin=dict(l=10, r=60, t=10, b=10), height=260,
                xaxis=dict(gridcolor=colors["border"], title="Contribution %"),
                yaxis=dict(gridcolor=colors["border"]),
                showlegend=False,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            if so_metrics:
                st.caption(
                    f"Model trained on {so_metrics['train_size']:,} samples | "
                    f"CV F1: {so_metrics['cv_f1_mean']:.2f} | "
                    f"Base OOS rate: {so_metrics['positive_rate']:.1%}"
                )
        else:
            st.info("Not enough data to evaluate model performance.")
