import pandas as pd
import numpy as np
import pickle
import hopsworks
import json
from datetime import datetime
import os

# Load dataset from the correct folder
df = pd.read_csv('csv/hourly_aqi_data.csv')

# Detecting and Removing Outliers
def remove_outliers_iqr(df, columns):
    cleaned_df = df.copy()
    for col in columns:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]
    return cleaned_df.reset_index(drop=True)


cols_to_clean = ['aqi', 'temperature', 'humidity', 'pm1', 'pm10']
df = remove_outliers_iqr(df, cols_to_clean)

## Feature Engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Lag features
df['aqi_lag_1h'] = df['aqi'].shift(1)      # 1 hour ago 
df['aqi_lag_3h'] = df['aqi'].shift(3)      # 3 hours ago 
df['aqi_lag_6h'] = df['aqi'].shift(6)      # 6 hours ago
df['aqi_lag_12h'] = df['aqi'].shift(12)    # 12 hours ago
df['aqi_lag_24h'] = df['aqi'].shift(24)    # 24 hours ago
df['aqi_lag_48h'] = df['aqi'].shift(48)    # 48 hours ago
df['aqi_lag_72h'] = df['aqi'].shift(72)    # 72 hours ago

# Cyclical encoding
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Rolling means
df['aqi_rolling_mean_3h'] = df['aqi'].shift(1).rolling(window=3, min_periods=1).mean()
df['aqi_rolling_mean_6h'] = df['aqi'].shift(1).rolling(window=6, min_periods=1).mean()
df['aqi_rolling_mean_12h'] = df['aqi'].shift(1).rolling(window=12, min_periods=1).mean()
df['aqi_rolling_mean_24h'] = df['aqi'].shift(1).rolling(window=24, min_periods=1).mean()
df['aqi_rolling_mean_48h'] = df['aqi'].shift(1).rolling(window=48, min_periods=1).mean()
df['aqi_rolling_mean_72h'] = df['aqi'].shift(1).rolling(window=72, min_periods=1).mean()

df = df.dropna()
df = df.drop(['city', 'pm25', 'dominentpol'], axis = 1)


### Creating the Pipeline (Feature Engineering)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_features = [
    'temperature', 'humidity', 'pm1', 'pm10', 'hour', 'day', 'month', 'day_of_week', 'is_weekend',
    'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h', 'aqi_lag_24h', 'aqi_lag_48h', 'aqi_lag_72h', 'sin_hour', 'cos_hour', 
    'sin_dayofweek', 'cos_dayofweek', 'aqi_rolling_mean_3h', 'aqi_rolling_mean_6h', 'aqi_rolling_mean_12h', 
    'aqi_rolling_mean_24h', 'aqi_rolling_mean_48h', 'aqi_rolling_mean_72h'
]

# Numeric transformation: impute + scale
numeric_transformer = Pipeline([
    ('simple_imputer_numerical', SimpleImputer(strategy='mean')),
    ('scaling_numeric', StandardScaler())
])

feature_engineering = ColumnTransformer([
    ('numeric', numeric_transformer, numeric_features)
])

pipe = Pipeline([
    ('feature_engineering_Pipeline', feature_engineering)
])

pipe.fit(df)

features_scaled = pipe.fit_transform(df)
all_cols = numeric_features

# Convert to DataFrame
features_scaled_df = pd.DataFrame(features_scaled, columns=all_cols)

# Add target (aqi) and timestamp for reference
features_scaled_df['aqi'] = df['aqi'].values
features_scaled_df['timestamp'] = df['timestamp'].values


### Storing the Features to Hopswork

import hopsworks
api_key = os.environ.get("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value = api_key)
fs = project.get_feature_store()

# create or get a feature group object
fg = fs.get_or_create_feature_group(
    name="aqi_scaled_features_10pearls",
    version=1,
    description="Scaled + engineered AQI features used by models",
    primary_key=["timestamp"],   
    event_time="timestamp"       
)
fg.insert(features_scaled_df, wait=True)


### Fetch the Scaled Features from Hopswork to Train Models

project = hopsworks.login(api_key_value = api_key)
fs = project.get_feature_store()
fg = fs.get_feature_group(
    name="aqi_scaled_features_10pearls",
    version=1
)
df_scaled = fg.read()


### Train and compare the ML Models 

X = df_scaled.drop(['aqi', 'timestamp'], axis = 1)
y = df_scaled['aqi']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
svr = SVR()
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()


# Grid Search CV on XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0.1, 1.0, 10.0]
}

search_cv_xgb = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_xgb,
    scoring='r2',
    cv=5,
    n_jobs=-1
)
search_cv_xgb.fit(X_train, y_train)
best_params_xgb = search_cv_xgb.best_params_
best_xgb_model = XGBRegressor(**best_params_xgb)


# Grid Search CV on Support Vector Regressor
param_grid_svr = {
    'kernel': ['rbf', 'poly'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale', 'auto']
}

search_cv_svr = GridSearchCV(
    estimator=svr,
    param_grid=param_grid_svr,
    scoring='r2',
    cv=5,
    n_jobs=-1
)
search_cv_svr.fit(X_train, y_train)
best_params_svr = search_cv_svr.best_params_
best_svr_model = SVR(**best_params_svr)


# Grid Search CV on RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
    'max_samples': [0.25, 0.50, 0.75, None],
    'max_depth': [2, 5, 7]
}

search_cv_rf = GridSearchCV(estimator = rf, 
                            param_grid = param_grid_rf, 
                            scoring = 'r2', 
                            cv = 5, 
                            n_jobs = -1)

search_cv_rf.fit(X_train, y_train)
best_params_rf = search_cv_rf.best_params_
best_rf_model = RandomForestRegressor(**best_params_rf)


# Grid Search CV on GradientBoosting

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 4, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0]
}

search_cv_gb = GridSearchCV(estimator = gb, 
                            param_grid = param_grid_gb, 
                            scoring = 'r2', 
                            cv = 5, 
                            n_jobs = -1)

search_cv_gb.fit(X_train, y_train)
best_params_gb = search_cv_gb.best_params_
best_gb_model = GradientBoostingRegressor(**best_params_gb)


# Now Try multiple models with the best parameters

results = []

models = {
    "XG Boost": best_xgb_model,
    "SVM": best_svr_model,
    "Random Forest": best_rf_model,
    "Gradient Boosting": best_gb_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    print(f"{name}: R2 Score = {r2:.4f}")
    print(f"{name}: Mean Square Error = {mse:.4f}")
    print(f"{name}: Mean Absolute Error = {mae:.4f}")
    print("")


    results.append({
        "Name": name,
        "R2_Score": r2  
    })


# Find best model based on R2 score
best_model_info = max(results, key=lambda x: x["R2_Score"])
best_model_name = best_model_info["Name"]
best_model_to_save = models[best_model_name]
best_model_to_save


## Save the best model
import pickle 
pickle.dump(best_model_to_save, open("models/best_model.pkl", 'wb')) 


## Apply MultiOutputRegressor for Multi-step Forecasting

base_model = pickle.load(open("models/best_model.pkl", "rb"))
HORIZON = 72
df = df_scaled.sort_values("timestamp").reset_index(drop=True)        
FEATURES = X.columns.tolist()
Y = pd.DataFrame({
    f"y_plus_{h}": df['aqi'].shift(-h) for h in range(1, HORIZON + 1)
})
valid_idx = Y.dropna().index
X_feat = df.loc[valid_idx, FEATURES].reset_index(drop=True)           
Y = Y.loc[valid_idx].reset_index(drop=True)                          


## Train the model
from sklearn.multioutput import MultiOutputRegressor

mor = MultiOutputRegressor(base_model)
mor.fit(X_feat, Y)
mor.feature_names_in_ = X_feat.columns.tolist()

# Save the model (MultiOutputRegressor)
pickle.dump(mor, open("models/best_forecast_model.pkl", "wb"))


## Save the Model to Hopswork 

project = hopsworks.login(api_key_value = api_key)
mr = project.get_model_registry()

best_model = pickle.load(open("models/best_model.pkl", "rb"))
best_forecast_model = pickle.load(open("models/best_forecast_model.pkl", "rb"))


# Register base model (single-step)
model_base = mr.python.create_model(
    name="aqi_base_model",
    description="Best single-step AQI prediction model (trained on scaled features)"
)
model_base.save("models/best_model.pkl")


# Register forecast model
model_forecast = mr.python.create_model(
    name="aqi_forecast_model",
    description="Multi-output 24-hour AQI forecasting model"
)
model_forecast.save("models/best_forecast_model.pkl")