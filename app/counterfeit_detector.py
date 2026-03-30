"""
Counterfeit Packaging Detector — Streamlit module
==================================================
Uses Google Gemini Vision API to analyze medicine packaging photos
for signs of counterfeiting in Pakistani pharmaceuticals.
"""

import os
import re
import hashlib
import base64

import streamlit as st
import requests as _req


def _get_gemini_key():
    """Get Gemini API key from secrets > env."""
    try:
        return st.secrets["GEMINI_API_KEY"].strip()
    except (KeyError, FileNotFoundError, AttributeError):
        return os.environ.get("GEMINI_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "PASS": "#22c55e",
    "WARNING": "#f59e0b",
    "FAIL": "#ef4444",
}

_RISK_COLORS = {
    "LOW_RISK": "#22c55e",
    "MEDIUM_RISK": "#f59e0b",
    "HIGH_RISK": "#ef4444",
}

_RISK_LABELS = {
    "LOW_RISK": "LOW RISK",
    "MEDIUM_RISK": "MEDIUM RISK",
    "HIGH_RISK": "HIGH RISK",
}

_PROMPT = """You are a pharmaceutical packaging quality inspector specializing in Pakistani medicines.

PART 1 — VISUAL INSPECTION
Analyze this medicine packaging image for signs of counterfeiting. Check each of the following and give a PASS, WARNING, or FAIL verdict for each:

1. DRAP REGISTRATION: Is a Drug Regulatory Authority of Pakistan (DRAP) registration number visible? (Format: "Reg. No. XXXXX")
2. BATCH & EXPIRY: Are batch number and expiry date clearly printed and not tampered?
3. MANUFACTURER INFO: Is manufacturer name, address, and country clearly printed?
4. TEXT QUALITY: Any spelling errors, inconsistent fonts, blurry text, or poor print quality?
5. LOGO & BRANDING: Does the logo/brand artwork appear professional? Any pixelation or color issues?
6. SECURITY FEATURES: Any holograms, QR codes, or security seals visible?
7. PACKAGING QUALITY: Does the material look professional (proper blister packs, quality cardboard)?
8. PRICE MARKING: Is Maximum Retail Price (MRP) printed?

For each check respond EXACTLY in this format:
CHECK_NAME: PASS|WARNING|FAIL - Brief explanation

Then on a new line:
OVERALL: LOW_RISK|MEDIUM_RISK|HIGH_RISK
CONFIDENCE: XX%
RECOMMENDATION: One-line recommendation

PART 2 — DATA EXTRACTION
Extract the following from the packaging image. If not visible, write "NOT_FOUND".
MEDICINE_NAME: <exact medicine name as printed on the box>
MRP_PRICE: <number only, in PKR, e.g. 150.00>
MANUFACTURER: <manufacturer name>
REG_NUMBER: <DRAP registration number if visible>"""

# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_CHECK_NAMES_MAP = {
    "DRAP REGISTRATION": "DRAP Registration",
    "DRAP_REGISTRATION": "DRAP Registration",
    "BATCH & EXPIRY": "Batch & Expiry",
    "BATCH_&_EXPIRY": "Batch & Expiry",
    "BATCH &amp; EXPIRY": "Batch & Expiry",
    "MANUFACTURER INFO": "Manufacturer Info",
    "MANUFACTURER_INFO": "Manufacturer Info",
    "TEXT QUALITY": "Text Quality",
    "TEXT_QUALITY": "Text Quality",
    "LOGO & BRANDING": "Logo & Branding",
    "LOGO_&_BRANDING": "Logo & Branding",
    "LOGO &amp; BRANDING": "Logo & Branding",
    "SECURITY FEATURES": "Security Features",
    "SECURITY_FEATURES": "Security Features",
    "PACKAGING QUALITY": "Packaging Quality",
    "PACKAGING_QUALITY": "Packaging Quality",
    "PRICE MARKING": "Price Marking",
    "PRICE_MARKING": "Price Marking",
}


def parse_counterfeit_response(raw_text: str) -> dict:
    """Parse the Gemini response into structured data."""
    result = {
        "checks": [],
        "overall_risk": "MEDIUM_RISK",
        "confidence": 0,
        "recommendation": "",
        "extracted": {},
        "raw": raw_text,
    }

    # Extract individual checks
    check_pattern = re.compile(
        r"([A-Z][A-Z &_]+?):\s*(PASS|WARNING|FAIL)\s*[-\u2013\u2014]\s*(.+)",
        re.IGNORECASE,
    )
    for match in check_pattern.finditer(raw_text):
        raw_name = match.group(1).strip().upper()
        status = match.group(2).strip().upper()
        explanation = match.group(3).strip()

        if raw_name in ("OVERALL", "CONFIDENCE", "RECOMMENDATION",
                        "MEDICINE_NAME", "MEDICINE NAME", "MRP_PRICE", "MRP PRICE",
                        "MANUFACTURER", "REG_NUMBER", "REG NUMBER"):
            continue

        display_name = _CHECK_NAMES_MAP.get(raw_name, raw_name.title())
        result["checks"].append({
            "name": display_name,
            "status": status,
            "explanation": explanation,
        })

    # Extract overall risk
    overall_match = re.search(
        r"OVERALL:\s*(LOW_RISK|MEDIUM_RISK|HIGH_RISK)", raw_text, re.IGNORECASE,
    )
    if overall_match:
        result["overall_risk"] = overall_match.group(1).upper()

    # Extract confidence
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)\s*%?", raw_text, re.IGNORECASE)
    if conf_match:
        result["confidence"] = int(conf_match.group(1))

    # Extract recommendation
    rec_match = re.search(r"RECOMMENDATION:\s*(.+)", raw_text, re.IGNORECASE)
    if rec_match:
        result["recommendation"] = rec_match.group(1).strip()

    # --- Part 2: Extracted data ---
    name_match = re.search(r"MEDICINE[_ ]?NAME:\s*(.+)", raw_text, re.IGNORECASE)
    if name_match:
        val = name_match.group(1).strip()
        if val.upper() != "NOT_FOUND":
            result["extracted"]["medicine_name"] = val

    price_match = re.search(r"MRP[_ ]?PRICE:\s*([\d,.]+)", raw_text, re.IGNORECASE)
    if price_match:
        try:
            result["extracted"]["mrp_price"] = float(price_match.group(1).replace(",", ""))
        except ValueError:
            pass

    mfg_match = re.search(r"MANUFACTURER:\s*(.+)", raw_text, re.IGNORECASE)
    if mfg_match:
        val = mfg_match.group(1).strip()
        if val.upper() != "NOT_FOUND":
            result["extracted"]["manufacturer"] = val

    reg_match = re.search(r"REG[_ ]?NUMBER:\s*(.+)", raw_text, re.IGNORECASE)
    if reg_match:
        val = reg_match.group(1).strip()
        if val.upper() != "NOT_FOUND":
            result["extracted"]["reg_number"] = val

    return result


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def _call_gemini(img_bytes: bytes, gemini_key: str) -> str:
    """Send image to Gemini Vision API and return raw text response."""
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "contents": [{
            "parts": [
                {"text": _PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
            ]
        }]
    }

    last_error = None
    for model_name in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={gemini_key}"
        )
        try:
            resp = _req.post(url, json=payload, timeout=60)
        except _req.exceptions.RequestException as exc:
            last_error = str(exc)
            continue

        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        last_error = resp.text

        # On rate limit, extract wait time and tell user
        if resp.status_code == 429:
            wait_match = re.search(r"retry in ([\d.]+)s", resp.text)
            wait_secs = int(float(wait_match.group(1))) + 1 if wait_match else 30
            raise RuntimeError(
                f"RATE_LIMITED|{wait_secs}"
            )

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# ML Verification Layer
# ---------------------------------------------------------------------------

def run_ml_verification(extracted: dict) -> dict | None:
    """Cross-reference Gemini-extracted data against our DRAP database and ML models.

    Returns a dict with verification results, or None if no medicine name was extracted.
    """
    medicine_name = extracted.get("medicine_name")
    mrp_price = extracted.get("mrp_price")

    if not medicine_name:
        return None

    import pandas as pd
    from difflib import get_close_matches, SequenceMatcher

    # Load DRAP data
    drap_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "drap_prices.csv",
    )
    try:
        drap_df = pd.read_csv(drap_path)
    except Exception:
        return None

    drap_names = drap_df["name"].tolist()
    result = {
        "extracted_name": medicine_name,
        "extracted_price": mrp_price,
        "checks": [],
    }

    # --- Check 1: DRAP Database Match ---
    matches = get_close_matches(medicine_name, drap_names, n=1, cutoff=0.5)
    if matches:
        best_match = matches[0]
        score = SequenceMatcher(None, medicine_name.lower(), best_match.lower()).ratio()
        drap_row = drap_df[drap_df["name"] == best_match].iloc[0]
        drap_price = drap_row["drap_price_pkr"]
        generic = drap_row.get("generic_name", "")

        result["drap_match"] = best_match
        result["match_score"] = round(score, 2)
        result["drap_price"] = drap_price
        result["generic"] = generic

        if score >= 0.85:
            result["checks"].append({
                "name": "DRAP Database Match",
                "status": "PASS",
                "explanation": f"Matched to \"{best_match}\" ({score:.0%} confidence). "
                               f"Registered generic: {generic}.",
            })
        elif score >= 0.6:
            result["checks"].append({
                "name": "DRAP Database Match",
                "status": "WARNING",
                "explanation": f"Partial match to \"{best_match}\" ({score:.0%} confidence). "
                               f"Verify the exact product name.",
            })
        else:
            result["checks"].append({
                "name": "DRAP Database Match",
                "status": "FAIL",
                "explanation": f"Low confidence match ({score:.0%}). "
                               f"Medicine may not be DRAP-registered.",
            })

        # --- Check 2: Price Deviation from DRAP ---
        if mrp_price and drap_price > 0:
            deviation_pct = ((mrp_price - drap_price) / drap_price) * 100

            if abs(deviation_pct) <= 5:
                result["checks"].append({
                    "name": "Price vs DRAP",
                    "status": "PASS",
                    "explanation": f"MRP Rs {mrp_price:,.0f} is within 5% of DRAP price "
                                   f"Rs {drap_price:,.0f} ({deviation_pct:+.1f}%).",
                })
            elif deviation_pct > 5 and deviation_pct <= 20:
                result["checks"].append({
                    "name": "Price vs DRAP",
                    "status": "WARNING",
                    "explanation": f"MRP Rs {mrp_price:,.0f} is {deviation_pct:.1f}% above DRAP price "
                                   f"Rs {drap_price:,.0f}. Slightly overpriced.",
                })
            elif deviation_pct > 20:
                result["checks"].append({
                    "name": "Price vs DRAP",
                    "status": "FAIL",
                    "explanation": f"MRP Rs {mrp_price:,.0f} is {deviation_pct:.1f}% above DRAP price "
                                   f"Rs {drap_price:,.0f}. Significant overpricing — possible counterfeit.",
                })
            elif deviation_pct < -30:
                result["checks"].append({
                    "name": "Price vs DRAP",
                    "status": "FAIL",
                    "explanation": f"MRP Rs {mrp_price:,.0f} is {abs(deviation_pct):.1f}% BELOW DRAP price "
                                   f"Rs {drap_price:,.0f}. Suspiciously cheap — high counterfeit risk.",
                })
            else:
                result["checks"].append({
                    "name": "Price vs DRAP",
                    "status": "PASS",
                    "explanation": f"MRP Rs {mrp_price:,.0f} vs DRAP Rs {drap_price:,.0f} ({deviation_pct:+.1f}%). "
                                   f"Price is within expected range.",
                })

        # --- Check 3: Isolation Forest Anomaly Detection ---
        if mrp_price and drap_price > 0:
            try:
                from app.ml_models import train_anomaly_model
                # Load live data
                import sqlite3
                db_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "medicines.db",
                )
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM prices ORDER BY scraped_at DESC", conn)
                conn.close()

                a_model, a_scaler, a_metrics = train_anomaly_model(df)
                if a_model and a_scaler:
                    import numpy as np
                    overprice_pct = ((mrp_price - drap_price) / drap_price) * 100
                    features = np.array([[mrp_price, overprice_pct, drap_price]])
                    scaled = a_scaler.transform(features)
                    prediction = a_model.predict(scaled)[0]
                    anomaly_score = -a_model.decision_function(scaled)[0]

                    if prediction == -1:
                        result["checks"].append({
                            "name": "ML Anomaly Detection",
                            "status": "FAIL",
                            "explanation": f"Isolation Forest flags this price as anomalous "
                                           f"(score: {anomaly_score:.3f}). "
                                           f"Price deviates significantly from market patterns.",
                        })
                    else:
                        result["checks"].append({
                            "name": "ML Anomaly Detection",
                            "status": "PASS",
                            "explanation": f"Price falls within normal market distribution "
                                           f"(anomaly score: {anomaly_score:.3f}).",
                        })
                    result["anomaly_score"] = round(float(anomaly_score), 3)
            except Exception:
                pass

        # --- Check 4: Market Price Comparison ---
        try:
            import sqlite3
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "medicines.db",
            )
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM prices ORDER BY scraped_at DESC", conn)
            conn.close()

            market_meds = df[df["name"].str.lower() == best_match.lower()]
            if not market_meds.empty:
                latest = market_meds.sort_values("scraped_at").drop_duplicates(
                    subset=["source"], keep="last"
                )
                market_avg = latest["price_pkr"].mean()
                market_min = latest["price_pkr"].min()
                market_max = latest["price_pkr"].max()

                result["market_avg"] = round(market_avg, 2)
                result["market_range"] = (round(market_min, 2), round(market_max, 2))

                if mrp_price:
                    market_dev = ((mrp_price - market_avg) / market_avg) * 100
                    if abs(market_dev) <= 10:
                        result["checks"].append({
                            "name": "Market Price Check",
                            "status": "PASS",
                            "explanation": f"MRP Rs {mrp_price:,.0f} aligns with market average "
                                           f"Rs {market_avg:,.0f} (range Rs {market_min:,.0f}-{market_max:,.0f}).",
                        })
                    elif market_dev > 10:
                        result["checks"].append({
                            "name": "Market Price Check",
                            "status": "WARNING",
                            "explanation": f"MRP Rs {mrp_price:,.0f} is {market_dev:.0f}% above market average "
                                           f"Rs {market_avg:,.0f}.",
                        })
                    elif market_dev < -25:
                        result["checks"].append({
                            "name": "Market Price Check",
                            "status": "FAIL",
                            "explanation": f"MRP Rs {mrp_price:,.0f} is {abs(market_dev):.0f}% below market average "
                                           f"Rs {market_avg:,.0f}. Suspiciously underpriced.",
                        })
                    else:
                        result["checks"].append({
                            "name": "Market Price Check",
                            "status": "PASS",
                            "explanation": f"MRP Rs {mrp_price:,.0f} vs market Rs {market_avg:,.0f} ({market_dev:+.0f}%).",
                        })
        except Exception:
            pass

    else:
        result["checks"].append({
            "name": "DRAP Database Match",
            "status": "FAIL",
            "explanation": f"\"{medicine_name}\" not found in DRAP database. "
                           f"Medicine may be unregistered or the name was misread.",
        })

    # Overall ML verdict
    statuses = [c["status"] for c in result["checks"]]
    fail_count = statuses.count("FAIL")
    warn_count = statuses.count("WARNING")

    if fail_count >= 2:
        result["ml_verdict"] = "HIGH_RISK"
    elif fail_count == 1 or warn_count >= 2:
        result["ml_verdict"] = "MEDIUM_RISK"
    else:
        result["ml_verdict"] = "LOW_RISK"

    return result


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _card_html(inner_html: str, extra_style: str = "") -> str:
    return (
        f'<div style="background:rgba(255,255,255,0.05);backdrop-filter:blur(16px);'
        f"border:1px solid rgba(255,255,255,0.08);border-radius:14px;"
        f'padding:1.25rem 1.5rem;margin-bottom:1rem;{extra_style}">'
        f"{inner_html}</div>"
    )


def _status_badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#71717a")
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:6px;'
        f"font-size:0.7rem;font-weight:600;letter-spacing:0.04em;"
        f'background:{color}22;color:{color};border:1px solid {color}44;">'
        f"{status}</span>"
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_counterfeit_detector_tab(colors):
    """Render the counterfeit packaging detector tab."""

    # ---- Header ----
    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(239,68,68,0.06) 0%, rgba(245,158,11,0.06) 50%, rgba(34,197,94,0.06) 100%);
                backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.06);
                border-radius:16px;padding:1.5rem 1.75rem;margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,0.4);"></div>
            <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#71717a;">AI-Powered Analysis</div>
        </div>
        <div style="font-size:1.2rem;font-weight:700;color:#fff;">Counterfeit Packaging Detector</div>
        <div style="font-size:0.8rem;color:#a1a1aa;margin-top:6px;line-height:1.5;">
            Upload a photo of your medicine's packaging. Our AI inspects 8 authenticity markers
            including DRAP registration, batch codes, print quality, and security features.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload area ----
    uploaded_file = st.file_uploader(
        "Upload medicine packaging image",
        type=["png", "jpg", "jpeg"],
        key="_cf_upload",
    )

    camera_input = None
    if st.checkbox("Use camera instead", key="_cf_use_camera"):
        camera_input = st.camera_input("Take a photo of the packaging", key="_cf_camera")

    img_source = uploaded_file or camera_input

    if not img_source:
        st.markdown(
            _card_html(
                '<div style="text-align:center;padding:2rem 1rem;">'
                '<div style="font-size:1.3rem;color:#71717a;margin-bottom:0.5rem;">Upload a photo</div>'
                '<div style="font-size:0.8rem;color:#52525b;">'
                "Take a photo or upload an image of your medicine's packaging to begin analysis."
                "</div></div>"
            ),
            unsafe_allow_html=True,
        )
        return

    # ---- Image preview ----
    st.image(img_source, caption="Medicine packaging", width=400)

    # ---- Demo mode toggle ----
    demo_mode = st.checkbox("Demo mode (skip API call, show sample result)", key="_cf_demo")

    if demo_mode:
        parsed = {
            "checks": [
                {"name": "DRAP Registration", "status": "WARNING", "explanation": "No clearly visible DRAP registration number found on the packaging."},
                {"name": "Batch & Expiry", "status": "PASS", "explanation": "Batch number and expiry date are clearly printed and appear legitimate."},
                {"name": "Manufacturer Info", "status": "PASS", "explanation": "Manufacturer name, address, and country of origin are clearly printed."},
                {"name": "Text Quality", "status": "PASS", "explanation": "Text is clear, consistent fonts, no spelling errors detected."},
                {"name": "Logo & Branding", "status": "PASS", "explanation": "Logo and brand artwork appear professional with consistent colors."},
                {"name": "Security Features", "status": "WARNING", "explanation": "No hologram or QR code visible on this side of the packaging."},
                {"name": "Packaging Quality", "status": "PASS", "explanation": "Professional quality cardboard and blister pack material."},
                {"name": "Price Marking", "status": "FAIL", "explanation": "No Maximum Retail Price (MRP) marking visible on the packaging."},
            ],
            "overall_risk": "MEDIUM_RISK",
            "confidence": 72,
            "recommendation": "Verify DRAP registration number with the manufacturer before use.",
            "extracted": {
                "medicine_name": "Panadol 500mg",
                "mrp_price": 38.00,
                "manufacturer": "GlaxoSmithKline",
            },
            "raw": "(demo mode)",
        }
        st.session_state["_cf_result"] = parsed
        st.session_state["_cf_img_hash"] = "demo"
    else:
        # ---- Gemini key ----
        gemini_key = _get_gemini_key()
        if not gemini_key:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                key="_cf_gemini_key",
                help="Get a free key at https://aistudio.google.com/apikey",
            )
        if not gemini_key:
            st.caption(
                "Enter a [free Gemini API key](https://aistudio.google.com/apikey) "
                "above to analyze the packaging."
            )
            return

        # ---- Session caching via MD5 ----
        img_source.seek(0)
        img_bytes = img_source.getvalue()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        if st.session_state.get("_cf_img_hash") != img_hash:
            status = st.empty()
            status.info("Analyzing packaging...")
            try:
                raw_text = _call_gemini(img_bytes, gemini_key)
                parsed = parse_counterfeit_response(raw_text)
                st.session_state["_cf_result"] = parsed
                st.session_state["_cf_img_hash"] = img_hash
                status.empty()
                st.rerun()
            except RuntimeError as exc:
                err_msg = str(exc)
                if err_msg.startswith("RATE_LIMITED|"):
                    wait_secs = int(err_msg.split("|")[1])
                    status.warning(f"Gemini rate limit hit. Wait ~{wait_secs}s then click **Retry**.")
                    if st.button("Retry Analysis", key="_cf_retry"):
                        st.session_state.pop("_cf_img_hash", None)
                        st.rerun()
                else:
                    status.error(f"Analysis failed: {err_msg}")
                st.stop()
            except Exception as exc:
                status.error(f"Unexpected error: {exc}")
                st.stop()

    parsed = st.session_state.get("_cf_result")
    if not parsed:
        return

    # ---- Risk Dashboard ----
    risk = parsed["overall_risk"]
    risk_color = _RISK_COLORS.get(risk, "#f59e0b")
    risk_label = _RISK_LABELS.get(risk, risk)
    confidence = parsed["confidence"]
    checks = parsed["checks"]

    pass_count = sum(1 for c in checks if c["status"] == "PASS")
    warn_count = sum(1 for c in checks if c["status"] == "WARNING")
    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    total_checks = max(len(checks), 1)

    # Confidence ring via SVG
    ring_pct = confidence / 100
    ring_dasharray = f"{ring_pct * 251.2:.1f} {251.2 - ring_pct * 251.2:.1f}"

    # ---- Risk verdict (use st.columns, not CSS grid) ----
    verdict_col, summary_col = st.columns([1, 2])

    with verdict_col:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid {risk_color}22;border-radius:16px;
                    padding:1.5rem 2rem;text-align:center;">
            <div style="position:relative;width:110px;height:110px;margin:0 auto 12px;">
                <svg viewBox="0 0 100 100" style="transform:rotate(-90deg);width:110px;height:110px;">
                    <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
                    <circle cx="50" cy="50" r="40" fill="none" stroke="{risk_color}" stroke-width="6"
                            stroke-dasharray="{ring_dasharray}" stroke-linecap="round"/>
                </svg>
                <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                            flex-direction:column;">
                    <div style="font-size:1.6rem;font-weight:800;color:{risk_color};line-height:1;">{confidence}%</div>
                    <div style="font-size:0.6rem;color:#71717a;margin-top:2px;">confidence</div>
                </div>
            </div>
            <div style="display:inline-block;padding:6px 18px;border-radius:8px;
                        background:{risk_color}15;border:1px solid {risk_color}30;
                        font-size:0.8rem;font-weight:700;letter-spacing:0.06em;color:{risk_color};">
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with summary_col:
        # Score bar
        total_width = pass_count + warn_count + fail_count or 1
        p_pct = pass_count / total_width * 100
        w_pct = warn_count / total_width * 100
        f_pct = fail_count / total_width * 100
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:16px;
                    padding:1.25rem 1.5rem;height:100%;">
            <div style="display:flex;gap:4px;height:10px;border-radius:5px;overflow:hidden;margin-bottom:10px;">
                <div style="width:{p_pct}%;background:#22c55e;"></div>
                <div style="width:{w_pct}%;background:#f59e0b;"></div>
                <div style="width:{f_pct}%;background:#ef4444;"></div>
            </div>
            <div style="display:flex;gap:1.5rem;margin-bottom:1rem;">
                <span style="font-size:0.8rem;color:#22c55e;font-weight:600;">{pass_count} passed</span>
                <span style="font-size:0.8rem;color:#f59e0b;font-weight:600;">{warn_count} warnings</span>
                <span style="font-size:0.8rem;color:#ef4444;font-weight:600;">{fail_count} failed</span>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:0.85rem 1rem;
                        border-left:3px solid {risk_color};">
                <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.04em;color:#52525b;margin-bottom:3px;">
                    Recommendation</div>
                <div style="font-size:0.85rem;color:#e4e4e7;">{parsed["recommendation"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Individual checks (use st.columns) ----
    status_icons = {"PASS": "&#10003;", "WARNING": "&#9888;", "FAIL": "&#10007;"}
    status_bg = {"PASS": "rgba(34,197,94,0.08)", "WARNING": "rgba(245,158,11,0.08)", "FAIL": "rgba(239,68,68,0.08)"}
    status_border_left = {"PASS": "#22c55e", "WARNING": "#f59e0b", "FAIL": "#ef4444"}

    if checks:
        for i in range(0, len(checks), 2):
            c1, c2 = st.columns(2)
            for col, check in zip([c1, c2], checks[i:i+2]):
                s = check["status"]
                sc = _STATUS_COLORS.get(s, "#71717a")
                col.markdown(f"""
                <div style="background:{status_bg.get(s, 'rgba(255,255,255,0.04)')};
                            border:1px solid rgba(255,255,255,0.06);border-left:3px solid {status_border_left.get(s, '#3f3f46')};
                            border-radius:10px;padding:0.85rem 1rem;margin-bottom:4px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <div style="font-size:0.85rem;font-weight:600;color:#e4e4e7;">{check['name']}</div>
                        <div style="display:flex;align-items:center;gap:5px;font-size:0.7rem;font-weight:600;
                                    color:{sc};letter-spacing:0.03em;">
                            <span style="font-size:0.85rem;">{status_icons.get(s, '')}</span>{s}
                        </div>
                    </div>
                    <div style="font-size:0.78rem;color:#a1a1aa;line-height:1.4;">{check['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;
                    padding:1rem 1.25rem;">
            <div style="font-size:0.7rem;text-transform:uppercase;color:#71717a;margin-bottom:6px;">Raw Analysis</div>
            <div style="font-size:0.85rem;color:#e4e4e7;white-space:pre-wrap;">{parsed["raw"]}</div>
        </div>""", unsafe_allow_html=True)

    # ---- ML Verification Layer ----
    extracted = parsed.get("extracted", {})
    if extracted:
        ml_result = run_ml_verification(extracted)
        if ml_result and ml_result["checks"]:
            ml_color = _RISK_COLORS.get(ml_result.get("ml_verdict", "MEDIUM_RISK"), "#f59e0b")
            ml_label = _RISK_LABELS.get(ml_result.get("ml_verdict", "MEDIUM_RISK"), "MEDIUM RISK")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(34,197,94,0.04) 100%);
                        border:1px solid rgba(255,255,255,0.06);border-radius:16px;
                        padding:1.25rem 1.5rem;margin:1.25rem 0;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#6366f1;box-shadow:0 0 8px rgba(99,102,241,0.4);"></div>
                    <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.08em;color:#71717a;">
                        ML Verification — Database Cross-Reference</div>
                </div>
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                    <div>
                        <div style="font-size:0.95rem;font-weight:600;color:#e4e4e7;">
                            Detected: <span style="color:#fff;">{ml_result.get('extracted_name', 'Unknown')}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#71717a;">
                            {'Matched to <strong>' + ml_result.get('drap_match', '') + '</strong> (' + ml_result.get('generic', '') + ')' if ml_result.get('drap_match') else 'No DRAP match found'}
                            {' &middot; DRAP Rs ' + f"{ml_result['drap_price']:,.0f}" if ml_result.get('drap_price') else ''}
                            {' &middot; Market avg Rs ' + f"{ml_result['market_avg']:,.0f}" if ml_result.get('market_avg') else ''}
                        </div>
                    </div>
                    <div style="display:inline-block;padding:4px 14px;border-radius:6px;
                                background:{ml_color}15;border:1px solid {ml_color}30;
                                font-size:0.7rem;font-weight:700;letter-spacing:0.05em;color:{ml_color};">
                        ML: {ml_label}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ML check cards
            for i in range(0, len(ml_result["checks"]), 2):
                mc1, mc2 = st.columns(2)
                for col, check in zip([mc1, mc2], ml_result["checks"][i:i+2]):
                    s = check["status"]
                    sc = _STATUS_COLORS.get(s, "#71717a")
                    s_icon = {"PASS": "&#10003;", "WARNING": "&#9888;", "FAIL": "&#10007;"}.get(s, "")
                    s_bg = {"PASS": "rgba(34,197,94,0.08)", "WARNING": "rgba(245,158,11,0.08)", "FAIL": "rgba(239,68,68,0.08)"}.get(s, "rgba(255,255,255,0.04)")
                    s_border = {"PASS": "#22c55e", "WARNING": "#f59e0b", "FAIL": "#ef4444"}.get(s, "#3f3f46")
                    col.markdown(f"""
                    <div style="background:{s_bg};border:1px solid rgba(255,255,255,0.06);
                                border-left:3px solid {s_border};border-radius:10px;
                                padding:0.85rem 1rem;margin-bottom:4px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                            <div style="font-size:0.85rem;font-weight:600;color:#e4e4e7;">{check['name']}</div>
                            <div style="display:flex;align-items:center;gap:5px;font-size:0.7rem;font-weight:600;
                                        color:{sc};letter-spacing:0.03em;">
                                <span style="font-size:0.85rem;">{s_icon}</span>{s}
                            </div>
                        </div>
                        <div style="font-size:0.78rem;color:#a1a1aa;line-height:1.4;">{check['explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    elif not demo_mode:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;padding:0.75rem 1rem;margin:1rem 0;">
            <div style="font-size:0.8rem;color:#71717a;">
                ML verification unavailable — could not extract medicine name from packaging.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Tips section ----
    with st.expander("How to verify your medicine"):
        t1, t2 = st.columns(2)
        t1.markdown("""
        <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.85rem 1rem;
                    border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
            <div style="font-size:0.8rem;font-weight:600;color:#e4e4e7;margin-bottom:4px;">Check DRAP Registration</div>
            <div style="font-size:0.75rem;color:#a1a1aa;">
                Visit <a href="https://www.dfrsa.gov.pk/" target="_blank" style="color:#71717a;">dfrsa.gov.pk</a>
                and search for the registration number on the pack.
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.85rem 1rem;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:0.8rem;font-weight:600;color:#e4e4e7;margin-bottom:4px;">Inspect Security Features</div>
            <div style="font-size:0.75rem;color:#a1a1aa;">
                Check for holograms, scratch-to-verify codes, or QR codes against the manufacturer's website.
            </div>
        </div>
        """, unsafe_allow_html=True)
        t2.markdown("""
        <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.85rem 1rem;
                    border:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
            <div style="font-size:0.8rem;font-weight:600;color:#e4e4e7;margin-bottom:4px;">Verify Batch Number</div>
            <div style="font-size:0.75rem;color:#a1a1aa;">
                Call the manufacturer's helpline and confirm the batch number matches their records.
            </div>
        </div>
        <div style="background:rgba(255,255,255,0.03);border-radius:10px;padding:0.85rem 1rem;
                    border:1px solid rgba(255,255,255,0.06);">
            <div style="font-size:0.8rem;font-weight:600;color:#e4e4e7;margin-bottom:4px;">Report Counterfeits</div>
            <div style="font-size:0.75rem;color:#a1a1aa;">
                DRAP helpline: <strong>0800-23727</strong> &middot; <strong>complaints@dfrsa.gov.pk</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
