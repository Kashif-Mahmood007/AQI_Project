# 🌿 Air Quality Index (AQI) Forecasting Project

This project analyzes and forecasts **Air Quality Index (AQI)** values to evaluate environmental conditions and predict pollution levels.
It leverages **machine learning**, **time series forecasting**, and **MLOps practices** to build an automated, end-to-end system — from **data ingestion to model deployment**.

---

## 🚀 Key Features

* **Automated Data Collection:** Fetches hourly AQI data from the WAQI API.
* **Data Preprocessing & Cleaning:** Handles missing values, outliers, and scaling.
* **Feature Engineering:** Creates lag features, rolling statistics, and cyclical encodings for improved temporal prediction.
* **Exploratory Data Analysis (EDA):** Visualizes AQI trends, distributions, and correlations.
* **Model Training & Evaluation:** Includes both base and multi-step forecasting models (up to 72 hours).
* **Model Registry:** Automatically stores models on **Hopsworks** for versioning and retrieval.
* **Frontend Dashboard:** Built using **Streamlit** to visualize real-time and forecasted AQI data.
* **CI/CD Pipeline:** Automated GitHub Actions for data fetching and daily model retraining.

---

## 🧩 Project Structure

```
AQI_Project/
│
├── csv/
│   └── hourly_aqi_data.csv                # Hourly AQI dataset
│
├── models/
│   ├── best_model.pkl                     # Base model for single-step prediction
│   ├── best_forecast_model.pkl            # Multi-step (72-hour) forecasting model
│   └── feature_order.json                 # Feature order reference (optional)
│
├── notebooks/
│   ├── Air Quality Index.ipynb            # Full workflow: feature engineering → model training → Hopsworks registry
│   └── Preprocessing & EDA.ipynb          # Data cleaning and exploratory analysis
│
├── src/
│   ├── app.py                             # Streamlit frontend
│   ├── fetch_aqi_data.py                  # Fetches AQI data hourly
│   └── train_model.py                     # Trains models daily and uploads to Hopsworks
│
├── venv/                                  # Python 3.11 virtual environment
│
├── .github/
│   └── workflows/
│       ├── fetch_aqi.yml                  # Runs hourly to collect new AQI data
│       └── train_model.yml                # Runs daily to retrain models
│
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Ignored files/folders
└── README.md                              # Project documentation
```

---

## 🧠 Requirements

* **Python 3.11**
* Install all required libraries from `requirements.txt`

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/<username>/AQI_Project.git
cd AQI_Project
```

2. **Create and Activate Virtual Environment**

```bash
py -3.11 -m venv venv
source venv/Scripts/activate   # Git Bash / Windows
# or
source venv/bin/activate       # macOS / Linux
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit App**

```bash
python src/app.py
```

---

## 🔐 GitHub Actions Setup

To enable automation, create **repository secrets**:

| Secret Name         | Description                              |
| ------------------- | ---------------------------------------- |
| `WAQI_TOKEN`        | Your WAQI API Key for fetching AQI data  |
| `HOPSWORKS_API_KEY` | Your Hopsworks API Key for model storage |

---

## 📊 Model Details

* **Base Model:** Utilize XGBRegressor, SVR (Support Vector Regressor), RandomForestRegressor, and GradientBoostingRegressor and select the best model among all. 
* **Forecasting Model:** Predicts AQI for the next **72 hours** using a MultiOutputRegressor
* **Model Storage:** Models are logged and versioned in **Hopsworks**.


---

## 👨‍💻 Author

**Kashif Mahmood**
Bachelor of Software Engineering | 10Pearls
💬 Passionate about **Data Science**, **Machine Learning**, and **MLOps**

---

## 🪴 License

This project is open-source and distributed under the **MIT License**.
