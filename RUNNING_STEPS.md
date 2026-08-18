# Running UBA Security Analytics Dashboard — Step-by-Step Guide

## Overview
This document provides a complete walkthrough of how to get the UBA (User Behaviour Analytics) Security Dashboard up and running from a fresh clone of the repository.

---

## Step 1: Project Structure Exploration

First, examine the project to understand what we're working with:

```
UBA_Final_Run/
├── data/
│   └── events.csv              # Sample behavioural events dataset
├── models/                     # Pre-trained models and artifacts
│   ├── CERT_performance.csv
│   ├── STUDY_ae.joblib         # Autoencoder model
│   ├── STUDY_cv.csv
│   ├── STUDY_feature_importance.csv
│   ├── STUDY_gb.joblib         # Gradient Boosting model
│   ├── STUDY_iso.joblib        # Isolation Forest model
│   ├── STUDY_metadata.json     # Model metadata (features, threshold)
│   ├── STUDY_performance.csv
│   ├── STUDY_preprocessor.joblib
│   └── STUDY_rf.joblib         # Random Forest model
├── dashboard.py                # Streamlit dashboard (main entry point)
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── README.md
└── Chapter_Four_Run_Report.txt
```

**Key insight**: The `models/` directory already contains pre-trained artifacts (`STUDY_*` and `CERT_*` files), so we can jump straight to running the dashboard without re-training.

---

## Step 2: Review Project Configuration Files

### 2.1 Read `requirements.txt`

The project requires these Python packages:

| Package            | Version    | Purpose                                        |
|--------------------|------------|------------------------------------------------|
| pandas             | >=2.0      | Data manipulation and CSV I/O                  |
| numpy              | >=1.24     | Numerical operations                           |
| scikit-learn       | >=1.3      | ML models (RF, GB, IsoForest) and metrics      |
| imbalanced-learn   | >=0.11     | SMOTE for class imbalance                      |
| joblib             | >=1.3      | Model serialization (load .joblib files)       |
| **streamlit**      | >=1.35     | **Dashboard framework (critical)**             |
| plotly             | >=5.18     | Interactive visualizations (bar charts, etc.)  |
| openpyxl           | >=3.1      | Excel file support                             |

### 2.2 Read `README.md`

The README documents three commands:
```bash
pip install -r requirements.txt
python train.py --data data/events.csv --target label --dataset CERT
streamlit run dashboard.py
```

Since models are already pre-trained, only the first and third commands are needed to run the dashboard.

---

## Step 3: Install Python Dependencies

### 3.1 Determine the correct Python command

On macOS, the system Python is often `python3` rather than `python`, and `pip3` rather than `pip`.

**Verification:**
```bash
python --version    # May fail: "command not found"
python3 --version   # Should work: Python 3.9.x
```

Similarly for pip:
```bash
pip --version       # May not be available
pip3 --version      # Should work
```

### 3.2 Run the dependency install

```bash
cd /path/to/UBA_Final_Run
pip3 install -r requirements.txt
```

**What happens during install:**
- ~40 packages are downloaded and installed (including transitive dependencies like `altair`, `pydeck`, `pyarrow`, `tornado`, etc.)
- On macOS with system Python, packages go to `~/Library/Python/3.9/lib/python/site-packages/`
- The `streamlit` executable is installed at `~/Library/Python/3.9/bin/streamlit`
- **Important**: This bin directory may NOT be on your `PATH`, so calling `streamlit` directly may fail with "command not found"

**Expected output** (abbreviated):
```
Defaulting to user installation because normal site-packages is not writeable
Collecting pandas>=2.0
  Downloading pandas-2.3.3-...
...
Successfully installed ... streamlit-1.50.0 ...

WARNING: The script streamlit is installed in '/Users/abraham/Library/Python/3.9/bin'
which is not on PATH.
```

### 3.3 Verify the installation

Because `streamlit` may not be on PATH, use the module invocation form:

```bash
python3 -m streamlit --version
```

Expected: `Streamlit, version 1.50.0` (or similar)

---

## Step 4: Understand the Dashboard Code (`dashboard.py`)

Before running, it's useful to understand what [dashboard.py](file:///Users/abraham/Documents/project/UBA_Final_Run/dashboard.py) does:

**Sections:**
1. **Imports** (line 3): `json`, `pathlib`, `joblib`, `numpy`, `pandas`, `streamlit`, `plotly.express`
2. **Page config** (line 6): Wide layout, title "UBA Security Analytics Dashboard"
3. **Load performance CSVs** (lines 10-13): Looks for `*_performance.csv` in `models/`. If missing, warns and stops.
4. **Dataset selector** (lines 15-17): Sidebar dropdown to switch between `STUDY` and `CERT` datasets.
5. **Best model metrics** (lines 19-22): 4-column KPIs — Best model name, Accuracy, F1, ROC-AUC.
6. **Model Comparison chart** (lines 24-27): Interactive Plotly bar chart. Select metric from dropdown.
7. **Confusion Matrix table** (lines 29-30): TN / FP / FN / TP per model.
8. **Detailed Performance table** (lines 32-33): Full performance dataframe.
9. **Feature Importance chart** (lines 35-39): Top 15 features (horizontal bar).
10. **Live Event Scoring** (lines 41-63):
    - Loads preprocessor + all 4 models (RF, GB, IsoForest, Autoencoder)
    - Dynamically generates numeric input fields for every feature (3-column layout)
    - "Analyse Event" button computes ensemble risk score:
      ```
      risk = 0.40*RF + 0.35*GB + 0.15*IsoForest + 0.10*Autoencoder
      ```
    - Compares against threshold from `metadata.json` → shows NORMAL (green) or ANOMALOUS (red)

---

## Step 5: Launch the Streamlit Dashboard

### 5.1 The correct launch command

Since the `streamlit` CLI may not be on PATH (see Step 3.2), launch via the Python module form:

```bash
cd /path/to/UBA_Final_Run
python3 -m streamlit run dashboard.py --server.headless true --server.port 8501
```

**Flags explained:**
- `run dashboard.py` — Specifies the Streamlit app file
- `--server.headless true` — Don't auto-open a browser window (useful for remote/headless environments)
- `--server.port 8501` — Bind to port 8501 (Streamlit default; explicit is clearer)
- `2>&1` (optional) — Merge stderr into stdout for consolidated logging

### 5.2 Expected startup output

```
Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.

/Users/abraham/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35:
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl'
module is compiled with 'LibreSSL 2.8.3'.
  warnings.warn(

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.0.161:8501
  External URL: http://102.88.110.20:8501
```

**Note on warnings**: The `NotOpenSSLWarning` from `urllib3` is cosmetic and does not affect functionality. The "Watchdog module" suggestion is also optional (only improves file-watching performance during development).

---

## Step 6: Access the Dashboard

Open any of these URLs in your web browser:

| URL Type     | Address                          | Use Case                               |
|--------------|----------------------------------|----------------------------------------|
| **Local**    | http://localhost:8501            | Accessing from the same machine        |
| Network      | http://192.168.0.161:8501        | Same LAN / same Wi-Fi network          |
| External     | http://102.88.110.20:8501        | Public internet (firewall permitting)  |

---

## Step 7: Use the Dashboard

### 7.1 Select a Dataset

From the **left sidebar**, choose:
- `STUDY` — Uses the STUDY_* model files (pre-trained on research data)
- `CERT` — Uses the CERT_* model files (pre-trained on CERT dataset)

### 7.2 Explore Model Comparison

1. At the top you'll see the 4 KPI cards:
   - **Best model** — Which model achieved the highest F1 score
   - **Accuracy** — Overall correct classification rate
   - **F1-score** — Harmonic mean of precision and recall
   - **ROC-AUC** — Area under receiver operating characteristic curve

2. In the **Model Comparison** section:
   - Use the dropdown to switch between `accuracy`, `precision`, `recall`, `f1`, `fpr`, `roc_auc`
   - Hover over bars for exact values

3. Scroll down to see:
   - **Confusion Matrix Components** (per model: tn, fp, fn, tp)
   - **Detailed Performance** (full table with all computed metrics)
   - **Behavioural Feature Importance** (top 15 features, horizontal bar)

### 7.3 Perform Live Event Scoring

1. Scroll to the **Live Behavioural Event Scoring** section at the bottom.
2. You will see numeric input fields for every feature listed in `models/{dataset}_metadata.json`, arranged in a 3-column grid.
3. Enter behavioural values for each feature (or leave at 0.0 for a baseline test).
4. Click the **"Analyse Event"** button (primary/blue).

**What happens next:**
1. The preprocessor transforms your raw inputs
2. All 4 models score the event independently
3. The ensemble formula produces a final risk score (0.0 — 1.0):
   ```
   risk = 0.40 * RF_prob
        + 0.35 * GB_prob
        + 0.15 * sigmoid(-5 * IsoForest_score)
        + 0.10 * sigmoid(-5 * AE_MSE)
   ```
4. A threshold comparison determines the verdict:
   - ✅ **NORMAL** (green success) — risk < threshold
   - ❌ **ANOMALOUS** (red error) — risk ≥ threshold → "send event to the approved security-review workflow"

---

## Step 8: (Optional) Retrain Models

If you want to retrain on new data instead of using the pre-trained artifacts:

```bash
cd /path/to/UBA_Final_Run
python3 train.py --data data/events.csv --target label --dataset CERT
```

This will:
1. Run end-to-end preprocessing + feature engineering
2. Apply SMOTE for class imbalance
3. Train 5 models (Rule-based, IsoForest, Autoencoder, RF, GBDT) + ensemble
4. Run 10-fold cross-validation
5. Compute all metrics, confusion matrices, and feature importance
6. Persist everything to `models/{dataset}_*` (overwriting existing ones)

After training completes, refresh the dashboard to see the new results.

---

## Troubleshooting

### Problem: `streamlit: command not found`

**Cause**: `~/Library/Python/3.9/bin` is not on `PATH`.

**Fix**: Use the module invocation form instead:
```bash
python3 -m streamlit run dashboard.py
```

Or add the directory to PATH:
```bash
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
streamlit run dashboard.py
```

---

### Problem: `ModuleNotFoundError: No module named 'streamlit'`

**Cause**: Dependencies were not installed, or installed for a different Python interpreter.

**Fix**:
```bash
pip3 install -r requirements.txt
python3 -m pip install -r requirements.txt   # alternative if pip3 is mismatched
```

Then verify:
```bash
python3 -c "import streamlit; print(streamlit.__version__)"
```

---

### Problem: Dashboard shows "No trained model results found"

**Cause**: Files matching `models/*_performance.csv` are missing.

**Fix**: Either:
- Ensure pre-trained model files exist in `models/` (check the download / unzip), OR
- Run `train.py` first (see Step 8 above) to regenerate them.

---

### Problem: `NotOpenSSLWarning` from urllib3 at startup

**Impact**: None — this is purely a warning, not an error.

**Suppress** (optional): Set before running:
```bash
export PYTHONWARNINGS="ignore::UserWarning::urllib3"
```

Or silence it programmatically in `dashboard.py` (after imports):
```python
import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
```

---

### Problem: Port 8501 is already in use

**Fix**: Use a different port:
```bash
python3 -m streamlit run dashboard.py --server.port 8502
```

Then access at http://localhost:8502

---

## Commands Summary (Quick Reference)

```bash
# 1. Install
cd /Users/abraham/Documents/project/UBA_Final_Run
pip3 install -r requirements.txt

# 2. Verify
python3 -m streamlit --version

# 3. Run dashboard
python3 -m streamlit run dashboard.py --server.headless true --server.port 8501

# 4. Open in browser → http://localhost:8501

# --- Optional: retrain models ---
python3 train.py --data data/events.csv --target label --dataset STUDY
python3 train.py --data data/events.csv --target label --dataset CERT
```
