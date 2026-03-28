# Pakistan Medicine Price Tracker (MedTracker PK)

A real-time medicine price monitoring dashboard that scrapes Pakistani pharmacy websites and compares prices against **DRAP (Drug Regulatory Authority of Pakistan)** registered rates to flag overpricing.

**[Live Demo](https://medtracker-pk.streamlit.app)**

---

## Why This Exists

Pakistan's pharmaceutical market serves over 230 million people, yet price transparency remains a critical challenge. Despite DRAP setting official Maximum Retail Prices (MRPs), many pharmacies — both online and offline — sell medicines above regulated prices. This disproportionately impacts low-income patients who already face barriers to healthcare access.

This project was built to:

- **Expose overpricing** by comparing real pharmacy prices against DRAP-registered rates
- **Empower consumers** with transparent, data-driven price comparisons
- **Support regulators** by providing evidence of pricing violations
- **Contribute to research** on healthcare accessibility in South Asia

> Healthcare is a right, not a privilege. Price transparency is the first step toward affordability.

---

## Features

### Medicine Lookup
- Search any medicine to find cheaper alternatives with the same active ingredient
- Best deal card with savings calculation
- Price comparison bar charts across pharmacies
- Direct links to Dawaai.pk and MedStore.com.pk

### Prescription Cart
- **AI Prescription Scanner** — Upload/photograph a prescription, Gemini AI extracts medicine names
- Fuzzy matching to find medicines even with typos
- Compare total prescription cost across pharmacies
- Identifies cheapest source with savings breakdown

### Cost Calculator
- Project monthly/multi-month medicine costs
- Automatic generic alternative suggestions with savings over time

### Pharmacies
- Interactive map of pharmacies across 6 major cities
- Filter by city, view address and contact details
- Online pharmacy links

### Analytics
- Overpricing analysis with top 10 overpriced medicines
- Availability tracking across pharmacies
- Price trend charts with DRAP reference lines
- Anomaly detection (IsolationForest) for unusual price spikes
- CSV and report export

### Research Insights
- **Medicine Accessibility Score** — per-city composite score (pharmacy density, price fairness, stock availability)
- **Affordability Index** — medicine costs as % of city-level income
- **City Affordability Heatmap** — essential medicines vs cities with income-adjusted burden
- **Pharmacy Coverage Map** — density analysis with Low/Medium/High coverage labels
- **Generic Penetration Analysis** — potential savings from switching to generics
- **WHO Essential Medicine Coverage** — cross-reference against 50 WHO essential medicines
- **Price & Overpricing Heatmaps** — visual price intensity and DRAP violation maps
- **Anomaly Detection Dashboard** — Isolation Forest scatter plot with detailed anomaly table
- **Stock-Out Tracker** — availability timeline heatmap for supply chain gap detection
- **Price Trend Forecasting** — linear projection with confidence bands

---

## Tech Stack

| Layer        | Technology                         |
|-------------|-------------------------------------|
| Frontend    | Streamlit, Plotly                   |
| Scraping    | Requests, BeautifulSoup, lxml       |
| Data        | Pandas, SQLite                      |
| ML          | scikit-learn (IsolationForest)      |
| AI/OCR      | Google Gemini API (prescription scanning) |
| Scheduling  | schedule (automated scraping)       |

---

## How to Run Locally

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/rushammm/Pakistan-Medicine-Tracker.git
cd Pakistan-Medicine-Tracker

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the scraper to populate the database
python scraper/scrape.py

# 5. Launch the dashboard
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** On the first run, the app will automatically generate synthetic data if the database is empty, so you can skip step 4 if you prefer.

### Prescription Scanner Setup (Optional)

To enable AI-powered prescription scanning, add your Gemini API key:

```bash
# Create .streamlit/secrets.toml
mkdir .streamlit
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### Scheduled Scraping

```bash
# Run every 6 hours (default)
python scraper/scrape.py --schedule

# Run every 2 hours
python scraper/scrape.py --schedule --interval 2
```

---

## Project Structure

```
pakistan-medicine-tracker/
├── app/
│   ├── app.py                  # Main Streamlit dashboard
│   └── research_insights.py    # Research Insights tab (10 features)
├── scraper/
│   └── scrape.py               # Web scraper + synthetic data generator
├── data/
│   ├── drap_prices.csv         # DRAP reference prices (183 medicines)
│   ├── pharmacies.csv          # Pharmacy locations (24 across 6 cities)
│   └── medicines.db            # SQLite database (auto-generated)
├── tests/
│   └── test_scanner.py         # Unit tests
├── .streamlit/
│   └── secrets.toml            # API keys (gitignored)
├── requirements.txt
├── runtime.txt                 # Python version for deployment
└── README.md
```

---

## Related Research

This project is inspired by and extends research on healthcare accessibility in South Asia:

**ICONIP 2024 Published Paper**
*"Healthcare Inaccessibility in South Asia: Challenges, Data-Driven Insights, and Pathways to Equitable Access"*
[Read the paper](https://arxiv.org/abs/2409.14194)

The Research Insights tab directly implements the paper's framework — using web-scraped pharmaceutical data, anomaly detection (Isolation Forest), accessibility scoring, and affordability analysis to identify and quantify healthcare gaps in Pakistan's medicine supply chain.

---

## How to Contribute

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

### Ideas for contributions

- Add more pharmacy websites as scraping sources
- Add SMS/email alerts for overpriced medicines
- Build a REST API layer for mobile apps
- Add Urdu language support
- Improve scraping resilience with Selenium/Playwright

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built by Rusham Elahi</strong><br>
  <em>Making medicine prices transparent in Pakistan</em>
</p>
