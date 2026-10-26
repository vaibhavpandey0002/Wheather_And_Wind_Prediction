<<<<<<< HEAD
# WeatherIQ — Intelligent Weather Temperature Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Frontend](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Visuals](https://img.shields.io/badge/Charts-Plotly-brightgreen.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-success.svg)](https://pytest.org/)

**WeatherIQ** is a production-quality college Machine Learning project that predicts future atmospheric temperatures using chronological weather observations. It pairs a supervised regression pipeline with modern meteorological analytics and presents everything through a SaaS-inspired dark navy analytics dashboard.

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Key Objectives & Features](#-key-objectives--features)
3. [Technology Stack & System Architecture](#-technology-stack--system-architecture)
4. [Dataset Requirements & Format](#-dataset-requirements--format)
5. [Installation & Windows PowerShell Setup](#-installation--windows-powershell-setup)
6. [Running the Application](#-running-the-application)
7. [Command-Line Model Training](#-command-line-model-training)
8. [Automated Test Suite](#-automated-test-suite)
9. [Machine Learning Pipeline Explained](#-machine-learning-pipeline-explained)
10. [Evaluation Metrics & The Persistence Baseline](#-evaluation-metrics--the-persistence-baseline)
11. [Why Synthetic Data Cannot Validate Real-World Forecasting](#-why-synthetic-data-cannot-validate-real-world-forecasting)
12. [College Demonstration & Viva Presentation Guide](#-college-demonstration--viva-presentation-guide)
13. [Limitations & Future Enhancements](#-limitations--future-enhancements)

---

## 🎯 Project Overview & Problem Statement

Short-term temperature forecasting is vital for renewable solar/wind grid stability, precision agriculture, and smart city HVAC automation. While complex numerical weather prediction (NWP) models require supercomputers, localized short-horizon temperature shifts can be accurately modeled using machine learning trained on sequence observations of:
1. **Previous Observed Temperature ($T_t$)** [°C]
2. **Relative Humidity ($RH_t$)** [%]
3. **Wind Speed ($WS_t$)** [km/h]

The target is the **future temperature** ($T_{t+h}$) at a configurable horizon (default $h = 1$ hour ahead).

---

## ✨ Key Objectives & Features

- **Real Machine Learning Pipeline:** Compares Linear Regression, Random Forest, and HistGradientBoosting against a naive **Persistence Baseline**.
- **No Data Leakage:** Implements strict chronological 3-way time-series partitioning (Train 70%, Validation 15%, Test 15%). No random shuffle leakage.
- **Physical Boundary Validation:** Powered by Pydantic v2 to catch sensor anomalies or impossible values (-60°C to +65°C, 0–100% RH, non-negative wind).
- **SQLite Prediction History:** Logs every inference with timestamps, inputs, predictions, model hashes, and observation notes.
- **Interactive Plotly Visualizations:** Temperature trajectories, moisture/wind dual-axis dynamics, diurnal 24-hour cycles, and residual error distributions.
- **Flexible Data Sources:** Upload custom historical CSV files, sync from the Open-Meteo Historical Archive, or generate deterministic demonstration data.
- **Audited Preprocessing:** Safe duplicate consolidation, chronological sorting, and time-based interpolation without silent mass row deletion.
- **Unit Conversions:** Supports seamless toggling between Celsius (°C) and Fahrenheit (°F) in the UI while preserving internal scientific integrity in °C.

---

## 🏗️ Technology Stack & System Architecture

### Core Technologies
- **Language:** Python 3.11+
- **Data Manipulation:** Pandas 2.x, NumPy 1.24+ / 2.x
- **Machine Learning:** Scikit-learn (Linear Regression, Random Forest, HistGradientBoosting, Pipelines, StandardScaler)
- **Serialization:** Joblib & JSON metadata
- **Persistence:** Local SQLite3 with indexed queries
- **Input Validation:** Pydantic v2
- **Dashboard:** Streamlit Multipage Framework
- **Data Visualization:** Plotly Graph Objects & Express
- **Testing:** Pytest

### Directory Structure
```
weatheriq/
├── app.py                      # Main application entry point & portal
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment configurations
├── .gitignore                  # Git ignore rules
├── README.md                   # Complete documentation & project report
│
├── pages/                      # Streamlit Multipage navigation
│   ├── 1_Overview.py           # Meteorological KPIs & telemetry dashboard
│   ├── 2_Prediction.py         # Manual parameter inference engine
│   ├── 3_Analytics.py          # Exploratory time-series & correlation suite
│   ├── 4_Model_Performance.py  # Held-out evaluation, benchmarks & residuals
│   ├── 5_Prediction_History.py # Local SQLite history with search & CSV export
│   ├── 6_Dataset_Management.py # CSV upload, Open-Meteo sync & model training
│   └── 7_Settings_About.py     # Global preferences & academic viva notes
│
├── src/                        # Modular backend & ML packages
│   ├── __init__.py             # Package version definition
│   ├── config.py               # Paths, theme tokens, physical boundaries
│   ├── data_loader.py          # CSV loading & column mapping heuristics
│   ├── data_validation.py      # Pydantic schemas & dataset validation rules
│   ├── preprocessing.py        # Cleaning, duplicate merging & interpolation
│   ├── feature_engineering.py  # Leakage-free chronological features & targets
│   ├── evaluate.py             # MAE, RMSE, R², and Persistence Baseline
│   ├── train.py                # Model training, model selection & CLI runner
│   ├── predict.py              # Inference service & rationale generator
│   ├── database.py             # SQLite CRUD operations
│   ├── weather_api.py          # Open-Meteo API historical archive client
│   ├── charts.py               # Dark-navy Plotly figure builders
│   ├── ui_helpers.py           # Shared cards, badges, and layout helpers
│   └── utils.py                # Conversions (°C/°F) and CSV encoders
│
├── assets/
│   └── styles.css              # Custom SaaS dark-navy theme tokens
│
├── data/
│   ├── .gitkeep
│   ├── weather_sample.csv      # Calibrated 90-day hourly synthetic dataset
│   └── prediction_history.db   # Local SQLite database
│
├── models/
│   ├── .gitkeep
│   ├── weather_model.joblib    # Active serialized Scikit-learn pipeline
│   └── model_metadata.json     # Hyperparameters, test metrics & data ranges
│
├── tests/                      # Automated Pytest suite
│   ├── __init__.py
│   ├── test_data_validation.py # Pydantic bounds & CSV schema checks
│   ├── test_preprocessing.py   # Chronological features & leakage prevention
│   ├── test_model.py           # Model training, inference & serialization
│   └── test_database.py        # SQLite insertion, filtering & deletion
│
└── scripts/
    └── generate_sample_data.py # Deterministic synthetic demonstration data
```

---

## 📊 Dataset Requirements & Format

A valid dataset requires chronological atmospheric observations with at least 48 records.

### Required Logical Columns:
1. **Timestamp:** Parseable datetime (e.g. `YYYY-MM-DD HH:MM:SS` or ISO-8601).
2. **Temperature:** Ambient air temperature in degrees Celsius (°C).
3. **Relative Humidity:** Value between 0.0% and 100.0%.
4. **Wind Speed:** Non-negative velocity in km/h.

### CSV Format Example:
```csv
timestamp,temperature,relative_humidity,wind_speed
2024-01-01 00:00:00,18.4,72.5,8.2
2024-01-01 01:00:00,17.9,74.1,7.5
2024-01-01 02:00:00,17.2,76.8,6.8
2024-01-01 03:00:00,16.8,78.2,6.1
2024-01-01 04:00:00,16.5,80.0,5.8
```

---

## 🚀 Installation & Windows PowerShell Setup

Open **Windows PowerShell** in the project directory:

```powershell
# 1. Create a Python virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip package installer
python -m pip install --upgrade pip

# 4. Install all dependencies
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web dashboard:

```powershell
streamlit run app.py
```

The interactive dashboard will open automatically in your browser at:
`http://localhost:8501`

---

## ⚙️ Command-Line Model Training

WeatherIQ provides a standalone CLI script to train models without launching the browser UI:

```powershell
# 1. Generate sample demonstration data (if needed)
python scripts/generate_sample_data.py

# 2. Train the model pipeline via CLI
python -m src.train --data data/weather_sample.csv --horizon 1
```

### CLI Parameters:
- `--data`: Path to the historical weather CSV (required).
- `--horizon`: Prediction lookahead in hours (default: 1).
- `--output-model`: Custom file path for the `.joblib` pipeline.
- `--output-meta`: Custom file path for the `.json` metadata.
- `--temp-col`, `--humidity-col`, `--wind-col`, `--timestamp-col`: Manual column overrides.

---

## 🧪 Automated Test Suite

Run all automated unit and integration tests with Pytest:

```powershell
python -m pytest tests/ -v
```

### Verified Test Matrix:
- `test_valid_prediction_input`: Validates physical inputs with Pydantic.
- `test_invalid_temperature_bounds`: Rejects out-of-bounds temperatures.
- `test_invalid_humidity_bounds`: Rejects humidity outside [0, 100]%.
- `test_invalid_wind_speed`: Rejects negative or excessive wind velocities.
- `test_csv_loading_from_buffer`: Verifies CSV buffer parsing.
- `test_missing_column_detection`: Detects missing mandatory columns.
- `test_duplicate_and_missing_timestamps`: Detects timestamp collisions and nulls.
- `test_clean_dataframe_preserves_zeros_and_sorts`: Ensures zero readings are preserved and nulls interpolated safely.
- `test_supervised_dataset_target_alignment_and_no_leakage`: Proves future data does not leak into input features.
- `test_chronological_train_test_split`: Confirms chronological order without random shuffling.
- `test_model_training_and_selection`: Trains candidates and compares test MAE.
- `test_prediction_output_validity`: Validates finite float prediction values.
- `test_missing_model_raises_file_not_found`: Gracefully handles absent model files.
- `test_db_init_and_insertion`: Tests SQLite schema creation and inserts.
- `test_db_deletion`: Tests record deletion and database purging.

---

## 🧠 Machine Learning Pipeline Explained

### 1. Mathematical Formulation
$$\hat{T}_{t+h} = f(T_t, RH_t, WS_t)$$
Where $T_t$ is observed temperature, $RH_t$ is relative humidity, and $WS_t$ is wind speed at time $t$. The target is temperature $h$ hours into the future.

### 2. Preventing Data Leakage
Standard cross-validation (like `KFold`) randomly shuffles observations. In time-series data, random shuffling causes severe temporal data leakage because consecutive weather readings are heavily autocorrelated. WeatherIQ strictly splits data in chronological order:
- **Training Set (70%):** Model fitting.
- **Validation Set (15%):** Hyperparameter comparison & model selection.
- **Held-Out Test Set (15%):** Untouched final evaluation benchmark.

Feature scaling (`StandardScaler`) is fitted strictly on training data inside a Scikit-learn `Pipeline` to prevent test-set distribution leakage.

---

## 📈 Evaluation Metrics & The Persistence Baseline

In regression modeling, metrics evaluate prediction error magnitude:
- **MAE (Mean Absolute Error):** Average magnitude of errors in degrees Celsius (°C). If MAE = 1.2°C, predictions deviate by 1.2°C on average.
- **RMSE (Root Mean Squared Error):** Penalizes large deviations heavily.
- **R² Score:** Proportion of variance explained by the model (1.0 is perfect; not an "accuracy percentage").

### The Persistence Baseline
Because weather changes smoothly over hours, a naive benchmark heuristic is the **Persistence Baseline**:
$$\hat{T}_{t+h}^{persistence} = T_t$$
It simply assumes tomorrow's or next hour's temperature equals current temperature. A machine learning model must prove it outperforms this baseline to justify its deployment.

---

## 🔬 Why Synthetic Data Cannot Validate Real-World Forecasting

WeatherIQ includes a deterministic demonstration dataset generator (`scripts/generate_sample_data.py`).
- **Purpose:** Enables offline installation, smoke testing, CI/CD pipelines, and college viva presentations without relying on live network APIs.
- **Scientific Limitation:** Synthetic data is modeled using mathematical formulas (sinusoidal curves + Gaussian noise). It **cannot** prove real-world meteorological predictive skill. For real scientific validation, always upload real station data or use the Open-Meteo archive connector in Dataset Management.

---

## 🎓 College Demonstration & Viva Presentation Guide

### Recommended Demonstration Script:
1. **Introduction:** Launch `streamlit run app.py` and explain the problem: predicting localized temperature 1-hour ahead.
2. **Architecture:** Point out the separation of concerns (`src/`, `pages/`, `data/`, `models/`).
3. **Data Quality & Ingestion:** Open **Dataset Management** (Page 6). Show automatic column mapping, missing value imputation, and out-of-bounds validation.
4. **Model Training & Baseline:** Show how the system fits Linear Regression, Random Forest, and HistGradientBoosting, selecting the best model based on held-out test MAE and beating the **Persistence Baseline**.
5. **Interactive Prediction:** Open **Temperature Prediction** (Page 2). Input 27.5°C, 52% humidity, 14 km/h wind. Show real-time inference, thermal shift rationale, and download the prediction CSV.
6. **Telemetry & Audit:** Open **Prediction History** (Page 5) to demonstrate local SQLite persistence. Show filtering, sorting, and single-record deletion.
7. **Analytics:** Open **Weather Analytics** (Page 3) to show the 24-hour diurnal cycle and feature cross-correlations.

---

## ⚖️ Limitations & Future Enhancements

### Current Limitations:
- Single-station local lookahead: Uses tabular regression rather than spatial grid maps.
- Exogenous variables: Solar irradiance, barometric pressure, and precipitation are not currently modeled.

### Future Enhancements:
- Multi-step autoregressive rollout (predicting 6, 12, and 24 hours sequentially).
- Deep learning sequence modeling (LSTM, Temporal Fusion Transformer).
- Integration of radar satellite imagery for cloud cover forecasting.

---

## 👨‍💻 Author & License
- **Project:** WeatherIQ — Intelligent Weather Temperature Prediction System
- **Purpose:** College Machine Learning Capstone / Semester Project
- **License:** MIT License
=======
# Wheather_And_Wind_Prediction
WeatherIQ is an end-to-end Machine Learning web application built with Python and Streamlit that predicts future atmospheric temperatures based on historical weather patterns (past temperature, humidity, and wind speed).
>>>>>>> 43c50e1be6166ce78cb6ea1bdaf14abc34ad69fe
