# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Fix random seed for reproducibility
np.random.seed(42)

# ==============================
# 1. Load Dataset
# ==============================
DATA_PATH = "data/weatherHistory.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully!")
print(df.head())
print(f"Dataset shape before cleaning: {df.shape}")

# ==============================
# 2. Basic Preprocessing
# ==============================
# Drop rows with missing values
df = df.dropna()

# Drop irrelevant text/categorical columns
drop_cols = ["Formatted Date", "Summary", "Precip Type", "Daily Summary"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

# Ensure target column exists
if "Temperature (C)" not in df.columns:
    raise KeyError("❌ Target column 'Temperature (C)' not found in dataset!")

# Features (X) and Target (y)
X = df.drop(columns=["Temperature (C)"])
y = df["Temperature (C)"]

print(f"Dataset shape after cleaning: X={X.shape}, y={y.shape}")

# ==============================
# 3. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Train Model
# ==============================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model training complete!")

# ==============================
# 5. Evaluate Model
# ==============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("📊 Model Performance:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")

# ==============================
# 6. Save Model and Feature Names
# ==============================
MODEL_PATH = "weather_model.pkl"
joblib.dump(
    {"model": model, "features": X.columns.tolist()},
    MODEL_PATH
)

print(f"✅ Model + feature names saved to {MODEL_PATH}")
