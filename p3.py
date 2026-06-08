import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("lungcancer.csv")

# Drop unused columns
drop_cols = [
    "index", "Patient Id", "Air Pollution", "OccuPational Hazards",
    "Balanced Diet", "Obesity", "Passive Smoker",
    "Clubbing of Finger Nails", "Frequent Cold",
    "Dry Cough", "Snoring"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Handle missing values: numeric -> mean, categorical -> mode
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Features & Target
X = df.drop(columns=["Level"])
y = df["Level"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Check if any NaNs remain
if X.isna().sum().sum() > 0:
    raise ValueError("NaNs still present in features!")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

# Train Logistic Regression (optional)
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_scaled, y_train)

# --- Save models, scaler, and feature columns ---
joblib.dump(rf, "models/lungcancer_rf_model.pkl")
joblib.dump(log_reg, "models/lungcancer_logreg_model.pkl")
joblib.dump(scaler, "models/lungcancer_scaler.pkl")
joblib.dump(X.columns, "models/lungcancer_features.pkl")

print("✅ Lung cancer models, scaler, and feature columns saved in 'models/' folder!")