"""
Price & Availability Intelligence
==================================
ML models trained on scraped pharmacy data to predict price movements
and monitor availability patterns.

Models:
- Random Forest: availability risk classifier (8 features)
- Gradient Boosting: price direction predictor (6 features)
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
# Core computations
# ---------------------------------------------------------------------------

def compute_all_predictions(df):
    """Batch-predict price direction and availability risk for all medicines."""
    so_model, so_le, so_metrics = train_stockout_model(df)
    p_model, p_le, p_metrics = train_price_model(df)

    if so_model is None:
        return [], None, None

    price_preds = predict_prices(p_model, p_le, df) if p_model else pd.DataFrame()

    results = []
    for med_name in sorted(df["name"].unique()):
        pred = predict_stockout(so_model, so_le, df, med_name)

        price_change = 0
        if not price_preds.empty:
            med_price = price_preds[price_preds["Medicine"] == med_name]
            if not med_price.empty:
                price_change = med_price["Change (%)"].iloc[0]

        med_df = df[df["name"] == med_name].sort_values("scraped_at")
        latest = med_df.iloc[-1] if not med_df.empty else None

        results.append({
            "medicine": med_name,
            "generic": latest.get("generic_name", "") if latest is not None else "",
            "current_price": latest["price_pkr"] if latest is not None else 0,
            "drap_price": latest.get("drap_price_pkr", 0) if latest is not None else 0,
            "source": latest.get("source", "") if latest is not None else "",
            "availability": latest.get("availability", "Unknown") if latest is not None else "Unknown",
            "risk_score": pred["risk_score"],
            "risk_level": pred["risk_level"],
            "price_change_pct": price_change,
        })

    return results, so_metrics, p_metrics


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_supply_forecaster_tab(df, colors, plotly_layout):
    """Render the Price & Availability Intelligence tab."""

    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(34,197,94,0.04) 100%);
                backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.06);
                border-radius:16px;padding:1.5rem 1.75rem;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#6366f1;box-shadow:0 0 8px rgba(99,102,241,0.4);"></div>
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#71717a;">Machine Learning</div>
        </div>
        <div style="font-size:1.2rem;font-weight:700;color:#fff;">Price & Availability Intelligence</div>
        <div style="font-size:0.8rem;color:#a1a1aa;margin-top:6px;line-height:1.5;">
            ML models trained on scraped data from dawaai.pk and dvago.pk to predict price
            movements and flag availability risks across tracked medicines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training models..."):
        results, so_metrics, p_metrics = compute_all_predictions(df)

    if not results:
        st.warning("Not enough historical data. Run the scraper a few more times.")
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
                    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.06em;color:#52525b;">Availability Model</div>
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

    # --- Price Predictions Table ---
    rising = [r for r in results if r["price_change_pct"] > 1]
    falling = [r for r in results if r["price_change_pct"] < -1]
    at_risk = [r for r in results if r["risk_level"] in ("High", "Medium")]

    # Compact summary
    st.markdown(f"""
    <div style="display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap;">
        <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:10px;
                    padding:0.6rem 1rem;flex:1;text-align:center;min-width:120px;">
            <div style="font-size:1.3rem;font-weight:700;color:#ef4444;">{len(rising)}</div>
            <div style="font-size:0.7rem;color:#ef4444;">prices predicted to rise</div>
        </div>
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:10px;
                    padding:0.6rem 1rem;flex:1;text-align:center;min-width:120px;">
            <div style="font-size:1.3rem;font-weight:700;color:#22c55e;">{len(falling)}</div>
            <div style="font-size:0.7rem;color:#22c55e;">prices predicted to drop</div>
        </div>
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:10px;
                    padding:0.6rem 1rem;flex:1;text-align:center;min-width:120px;">
            <div style="font-size:1.3rem;font-weight:700;color:#f59e0b;">{len(at_risk)}</div>
            <div style="font-size:0.7rem;color:#f59e0b;">availability concerns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Combined predictions table ---
    table_data = []
    for r in sorted(results, key=lambda x: x["price_change_pct"], reverse=True):
        pc = r["price_change_pct"]
        price_arrow = "+" if pc > 1 else ("-" if pc < -1 else "")
        risk_level = r["risk_level"]

        table_data.append({
            "Medicine": r["medicine"],
            "Generic": r["generic"],
            "Price (PKR)": f"Rs {r['current_price']:,.0f}",
            "Predicted Change": f"{price_arrow}{pc:.1f}%",
            "Availability Risk": risk_level,
            "Risk Score": f"{r['risk_score']:.0%}",
            "Source": r["source"],
            "_pc": pc,
            "_risk": r["risk_score"],
        })

    table_df = pd.DataFrame(table_data)
    display_df = table_df.drop(columns=["_pc", "_risk"])

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Predicted Change": st.column_config.TextColumn("Price Forecast (GB)"),
            "Availability Risk": st.column_config.TextColumn("Avail. Risk (RF)"),
        },
    )

    st.caption("Predictions from Random Forest (availability) and Gradient Boosting (price). Based on dawaai.pk and dvago.pk data only — not a nationwide forecast.")

    # --- Medicine Deep Dive ---
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Medicine Deep Dive</div>', unsafe_allow_html=True)

    all_meds = sorted(df["name"].unique())
    selected = st.selectbox("Select a medicine", all_meds, key="forecast_medicine_select")

    if selected:
        med_result = next((r for r in results if r["medicine"] == selected), None)
        if med_result:
            risk_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(med_result["risk_level"], "#71717a")
            pc = med_result["price_change_pct"]
            pc_color = "#ef4444" if pc > 1 else ("#22c55e" if pc < -1 else "#a1a1aa")
            pc_arrow = "&#9650;" if pc > 1 else ("&#9660;" if pc < -1 else "&#9644;")
            avail_color = {"In Stock": "#22c55e", "Limited": "#f59e0b", "Out of Stock": "#ef4444"}.get(med_result["availability"], "#71717a")

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                        border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                    <div>
                        <div style="font-size:1rem;font-weight:700;color:#fff;">{selected}</div>
                        <div style="font-size:0.8rem;color:#a1a1aa;">{med_result['generic']} &middot; Rs {med_result['current_price']:,.0f}
                            <span style="color:{avail_color};margin-left:6px;">&#9679; {med_result['availability']}</span>
                        </div>
                    </div>
                    <div style="display:flex;gap:1.5rem;">
                        <div style="text-align:center;">
                            <div style="font-size:0.6rem;text-transform:uppercase;color:#52525b;">Price Forecast</div>
                            <div style="font-size:1rem;font-weight:700;color:{pc_color};">{pc_arrow} {pc:+.1f}%</div>
                            <div style="font-size:0.55rem;color:#52525b;">Gradient Boosting</div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.6rem;text-transform:uppercase;color:#52525b;">Avail. Risk</div>
                            <div style="font-size:1rem;font-weight:700;color:{risk_color};">{med_result['risk_level']}</div>
                            <div style="font-size:0.55rem;color:#52525b;">RF {med_result['risk_score']:.0%}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Price trajectory chart
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
                margin=dict(l=10, r=10, t=30, b=10), height=260,
                title="Price History",
                xaxis=dict(gridcolor=colors["border"]),
                yaxis=dict(gridcolor=colors["border"], title="Price (PKR)"),
                hoverlabel=dict(bgcolor="#1a1a1f", bordercolor="#3f3f46", font_color="#e4e4e7"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Prophet Forecast ---
        try:
            from forecasting.price_forecast import forecast_price, get_forecast_insight

            history, forecast_df, data_count = forecast_price(selected, days=7)

            if forecast_df is not None and history is not None:
                current_price = history["y"].iloc[-1]
                insight = get_forecast_insight(forecast_df, current_price)

                # Forecast chart
                forecast_only = forecast_df[forecast_df["ds"] > history["ds"].max()]
                dir_color = "#ef4444" if insight["direction"] == "rising" else (
                    "#22c55e" if insight["direction"] == "falling" else "#a1a1aa"
                )

                fig_fc = go.Figure()
                # Historical
                fig_fc.add_trace(go.Scatter(
                    x=history["ds"], y=history["y"],
                    mode="lines+markers", name="Historical (real)",
                    line=dict(color="#fafafa", width=2), marker=dict(size=4),
                ))
                # Forecast
                if not forecast_only.empty:
                    fig_fc.add_trace(go.Scatter(
                        x=forecast_only["ds"], y=forecast_only["yhat"],
                        mode="lines+markers", name="Prophet forecast",
                        line=dict(color=dir_color, width=2, dash="dash"),
                        marker=dict(size=4),
                    ))
                    # Confidence band
                    fig_fc.add_trace(go.Scatter(
                        x=pd.concat([forecast_only["ds"], forecast_only["ds"][::-1]]),
                        y=pd.concat([forecast_only["yhat_upper"], forecast_only["yhat_lower"][::-1]]),
                        fill="toself", fillcolor=f"{dir_color}15",
                        line=dict(width=0), name="90% confidence",
                        showlegend=True,
                    ))
                    # Today line
                    fig_fc.add_vline(
                        x=history["ds"].max(), line_dash="dot", line_color="#71717a",
                        annotation_text="today", annotation_font_size=9,
                        annotation_font_color="#71717a",
                    )

                fig_fc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=colors["text"], size=11),
                    margin=dict(l=10, r=10, t=30, b=10), height=260,
                    title="Prophet 7-Day Price Forecast",
                    xaxis=dict(gridcolor=colors["border"]),
                    yaxis=dict(gridcolor=colors["border"], title="Price (PKR)"),
                    hoverlabel=dict(bgcolor="#1a1a1f", bordercolor="#3f3f46", font_color="#e4e4e7"),
                    legend=dict(font=dict(color=colors["text"], size=10)),
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                # Insight card
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                            border-left:3px solid {dir_color};border-radius:10px;padding:0.75rem 1rem;">
                    <div style="font-size:0.85rem;color:#e4e4e7;">{insight['message']}</div>
                    <div style="font-size:0.7rem;color:#52525b;margin-top:4px;">
                        Prophet model trained on {data_count} real data points</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                needed = max(0, 14 - data_count)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                            border-radius:10px;padding:0.75rem 1rem;margin-top:0.5rem;">
                    <div style="font-size:0.8rem;color:#71717a;">
                        Prophet forecast unavailable — only {data_count} data points
                        (need {14}). Run the scraper {needed} more times to unlock.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except ImportError:
            st.caption("Prophet not installed. Run `pip install prophet` to enable price forecasting.")
        except Exception:
            pass

    # --- Model Evaluation ---
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Model Evaluation</div>', unsafe_allow_html=True)
    with st.expander("ROC Curve, Confusion Matrix & Feature Importance", expanded=True):
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
                split_note = (
                    f"Temporal split at {eval_metrics.get('cutoff_date', 'N/A')[:10]}"
                    if eval_metrics and eval_metrics.get("split_method") == "temporal"
                    else "Train/test split"
                )
                st.caption(
                    f"{split_note} | "
                    f"Train: {eval_metrics['train_size']:,} · Test: {eval_metrics['test_size']:,} | "
                    f"Generic stats computed on train set only (no leakage)"
                )
        else:
            st.info("Not enough data to evaluate model performance.")
