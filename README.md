# MedTracker PK

Real-time pharmaceutical intelligence platform for Pakistan. Scrapes medicine prices from online pharmacies, compares against DRAP-regulated rates, and uses ML models to predict supply shortages and detect pricing anomalies.

**[Live Demo](https://medtracker-pk.streamlit.app)** | **[Research Paper (ICONIP 2024)](https://arxiv.org/abs/2409.14194)**

## What It Does

**Prescription Scanner** — Upload a prescription photo. Gemini Vision extracts medicine names, fuzzy-matches them against the database, and compares total cost across pharmacies. ML models flag any medicines at risk of going out of stock.

**Supply Shortage Forecaster** — Random Forest classifier (8 features, AUC ~0.73) predicts stock-out probability for each medicine. Gradient Boosting regressor predicts price direction. Combined into human-readable timelines: *"This medicine will be short within 2-4 weeks."*

**Counterfeit Packaging Detector** — Two-layer verification: Gemini Vision inspects 8 visual authenticity markers, then an ML pipeline cross-references the extracted medicine name and MRP against the DRAP database, runs Isolation Forest anomaly detection on the price, and compares against live market data.

**Supply Gap Analysis** — Cross-references scraped availability data against WHO essential medicines to identify which drugs face persistent supply constraints in Pakistan's online pharmacy infrastructure.

## ML Pipeline

| Model | Algorithm | Features | Target | Metric |
|-------|-----------|----------|--------|--------|
| Stock-out classifier | Random Forest (100 trees, depth 6) | price, DRAP price, overprice %, availability lag, source, price change, generic avg, brand count | Out of stock next scrape | F1, ROC AUC |
| Price predictor | Gradient Boosting (100 trees, lr 0.1) | price, DRAP price, overprice %, availability, medicine encoded, price change | Next-period price | R², CV R² |
| Anomaly detector | Isolation Forest (200 estimators) | price, overprice %, DRAP price | Pricing outlier | Contamination 0.1 |

Models train on real data scraped from **dawaai.pk** and **dvago.pk**, compared against **183 DRAP-registered medicine prices**.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly |
| ML | scikit-learn (RandomForest, GradientBoosting, IsolationForest) |
| AI/Vision | Google Gemini API (prescription OCR, packaging analysis) |
| Scraping | Requests, BeautifulSoup, lxml |
| Data | Pandas, SQLite |

## Data Sources

- **dawaai.pk** — scraped via sitemap + JSON-LD structured data
- **dvago.pk** — scraped via product API
- **DRAP** — official regulated prices (183 medicines)

## Research Connection

Extends research published at ICONIP 2024: *"Healthcare Inaccessibility in South Asia: Challenges, Data-Driven Insights, and Pathways to Equitable Access"* ([arXiv:2409.14194](https://arxiv.org/abs/2409.14194)). The supply gap analysis applies the paper's accessibility framework to real-time scraped pharmaceutical data.

<p align="center">
  <strong>Built by Rusham Elahi</strong>
</p>
