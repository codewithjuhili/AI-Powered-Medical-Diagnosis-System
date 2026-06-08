import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Starting diabetes model training...")

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
print("Loading dataset...")
data = pd.read_csv("diabetes.csv")
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# Identify categorical columns (exclude numeric columns)
numeric_cols = ["age", "bmi", "sleep_hours", "risk_score"]
categorical_cols = [col for col in data.columns if col not in numeric_cols]

# Remove target column from categorical columns
if "risk_label" in categorical_cols:
    categorical_cols.remove("risk_label")
    target_col = "risk_label"
elif "diabetes" in categorical_cols:
    categorical_cols.remove("diabetes")
    target_col = "diabetes"
else:
    raise ValueError("No target column found. Expected 'diabetes' or 'risk_label'")

# Encode categorical columns
encoders = {}
for col in categorical_cols:
    if col in data.columns:
        enc = LabelEncoder()
        data[col] = enc.fit_transform(data[col].astype(str).str.lower())
        encoders[col] = enc

# Save encoder for target label if it exists and is categorical
target_encoder = None
if target_col in data.columns:
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(data[target_col].astype(str).str.lower())
    encoders['risk_label'] = target_encoder  # Save for inverse transform later
    y = pd.Series(y_encoded)
else:
    raise ValueError(f"Target column '{target_col}' not found in data")

# Features & target
X = data.drop(target_col, axis=1)

# Handle NaN
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Save model + encoders + imputer + features
joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(encoders, "models/diabetes_encoders.pkl")
joblib.dump(imputer, "models/diabetes_imputer.pkl")
joblib.dump(X.columns, "models/diabetes_features.pkl")

print("✅ Diabetes model saved!")
print(f"   - Model: models/diabetes_model.pkl")
print(f"   - Encoders: models/diabetes_encoders.pkl")
print(f"   - Imputer: models/diabetes_imputer.pkl")
print(f"   - Features: models/diabetes_features.pkl")
print(f"\n📊 Model trained on {len(X_train)} samples with {len(X.columns)} features")
print(f"🎯 Target classes: {target_encoder.classes_}")
