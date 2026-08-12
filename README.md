# 🏠 Real Estate Appraisal Regression & Error Diagnostics Suite

An automated evaluation framework for housing price models. It goes beyond a single
R² score and shows you **MAE, RMSE, MAPE, and — the key feature — error broken down
by price bracket**, so a model that looks great on average but quietly fails on
expensive homes gets caught before it reaches production.

Dataset: **Ames Housing Dataset** (1,460 homes, 80 features, sale prices $34,900–$755,000).
The CSV is already included in `data/AmesHousing.csv` — no download needed.

---

## What's inside

```
appraisal_suite/
├── app.py                  # Streamlit dashboard (the main UI)
├── run_cli_report.py       # Optional: generates a static HTML report, no browser needed
├── requirements.txt        # Python dependencies
├── run_windows.bat         # One-click setup + launch on Windows
├── run_mac_linux.sh        # One-click setup + launch on Mac/Linux
├── data/
│   └── AmesHousing.csv     # The dataset (already included)
├── src/
│   └── pipeline.py         # All the ML logic: preprocessing, models, multi-split eval, metrics
└── outputs/                # Reports/exports land here
```

---

## Step-by-step setup

### Option A — Windows

1. Unzip this folder anywhere (e.g. Desktop).
2. Make sure [Python 3.9+](https://www.python.org/downloads/) is installed and on PATH.
3. Double-click **`run_windows.bat`**.
   - It creates a virtual environment, installs everything, and opens the dashboard
     in your browser automatically.

### Option B — Mac / Linux

1. Unzip the folder.
2. Open Terminal in that folder.
3. Run:
   ```bash
   chmod +x run_mac_linux.sh
   ./run_mac_linux.sh
   ```
4. Your browser opens automatically at `http://localhost:8501`.

### Option C — Manual setup (any OS)

```bash
cd appraisal_suite
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## How to use the dashboard

1. **Pick models** in the sidebar (Linear, Ridge, Lasso, Decision Tree, Random Forest,
   Gradient Boosting — mix and match).
2. **Set the number of splits** — the suite trains/tests each model that many times
   with different random train/test partitions, so results aren't a fluke of one split.
3. **Set test size and price brackets** (Low/Mid/High, or finer).
4. Click **🚀 Run Evaluation**.
5. Explore the tabs:
   - **Overview** — headline metrics + why R² alone can mislead you.
   - **Model Comparison** — MAE/RMSE/R² table, stability across splits (box plot).
   - **Price-Bracket Diagnostics** — the core feature: MAE/RMSE/MAPE per price tier,
     so you can see exactly where a model's accuracy breaks down.
   - **Residual Analysis** — predicted-vs-actual and residual scatter plots per model.
   - **Raw Results** — full per-split tables + CSV downloads.

## No-UI alternative

If you just want numbers/plots in a single file (e.g., to email or archive):

```bash
python run_cli_report.py --splits 5 --test-size 0.2 --brackets 3
```

This writes `outputs/diagnostics_report.html` — open it in any browser.

---

## Metrics explained

| Metric | What it tells you |
|---|---|
| **MAE** (Mean Absolute Error) | Average dollar error, in plain terms. |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalizes big misses harder — useful for catching a model that's occasionally *very* wrong. |
| **MAPE** (Mean Absolute % Error) | Error as a percentage of price — lets you compare accuracy on cheap vs. expensive homes fairly. |
| **R²** | Fraction of price variance explained. High R² can still hide large dollar errors on a subset of homes — that's exactly what the price-bracket tab is for. |
| **Max Error** | The single worst prediction in the test set. |

---

## Extending it

- Add a model: edit `get_model_zoo()` in `src/pipeline.py`.
- Change how price brackets are computed: edit `assign_price_brackets()`.
- Swap in a different dataset: replace `data/AmesHousing.csv` with any CSV that has a
  `SalePrice` column (adjust `TARGET` in `pipeline.py` if the column is named differently).
