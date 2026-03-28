# Pakistan Medicine Price Tracker (MedTracker PK)



A real-time medicine price monitoring dashboard that scrapes Pakistani pharmacy websites and compares prices against **DRAP (Drug Regulatory Authority of Pakistan)** registered rates to flag overpricing.

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

-  **Multi-Source Scraping** — Fetches prices from [Dawaai.pk](https://dawaai.pk) and [MedStore.com.pk](https://medstore.com.pk)
-  **DRAP Comparison** — Flags medicines priced >10% above official DRAP rates
-  **Anomaly Detection** — Uses scikit-learn IsolationForest to flag unusual price spikes
-  **Price Trends** — Tracks price changes over time with interactive line charts
-  **Visual Analytics** — Bar charts (top overpriced), pie charts (fair vs overpriced)
-  **Search & Filter** — Find any medicine instantly, filter by overpriced status or source
-  **CSV/Report Export** — Download filtered data as CSV or a summary report
-  **Scheduled Scraping** — Automated price monitoring on a configurable interval
-  **On-Demand Scraping** — Trigger a fresh scrape from the sidebar
-  **Synthetic Fallback** — Always shows data, even if live scraping is blocked
-  **SQLite Storage** — Persistent local database for historical tracking

---

## Tech Stack

| Layer        | Technology                         |
|-------------|-------------------------------------|
| Frontend    | Streamlit, Plotly                   |
| Scraping    | Requests, BeautifulSoup, lxml       |
| Data        | Pandas, SQLite                      |
| ML          | scikit-learn (IsolationForest)      |
| Scheduling  | schedule (automated scraping)       |
| Testing     | pytest                              |

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

### Scheduled Scraping

To run the scraper on an automatic schedule:

```bash
# Run every 6 hours (default)
python scraper/scrape.py --schedule

# Run every 2 hours
python scraper/scrape.py --schedule --interval 2
```

### Running Tests

```bash
python -m pytest tests/ -v
```

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

## Related Research

This project is inspired by and contributes to ongoing research on healthcare accessibility in South Asia:

 **ICONIP 2024 Published Paper**
*"Healthcare Inaccessibility in South Asia: Challenges, Data-Driven Insights, and Pathways to Equitable Access"*
[Read the paper →](https://link.springer.com/conference/iconip)

The paper explores systemic barriers to healthcare access across South Asian countries and proposes data-driven interventions — of which price transparency tools like this project are a practical implementation.

---

## Project Structure

```
pakistan-medicine-tracker/
├── scraper/
│   └── scrape.py           # Web scraper (dawaai.pk + medstore.com.pk) + anomaly detection
├── data/
│   ├── drap_prices.csv     # DRAP reference prices (25 medicines)
│   └── medicines.db        # SQLite database (auto-generated)
├── app/
│   └── app.py              # Streamlit dashboard with exports & anomaly markers
├── tests/
│   └── test_scraper.py     # Unit tests
├── .gitignore
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Built by Rusham Elahi</strong><br>
  <em>Making medicine prices transparent in Pakistan</em>
</p>
