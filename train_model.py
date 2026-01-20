import pandas as pd
import numpy as np
import re
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
import os
import pandas as pd

if not os.path.exists("laptop (3).csv"):
    url = "https://raw.githubusercontent.com/rajkumarv07/Laptop_Price-prediction/main/laptop%20(3).csv"
    df = pd.read_csv(url)
    df.to_csv("laptop (3).csv", index=False)
else:
    df = pd.read_csv("laptop (3).csv")


# Drop junk columns
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
df = df.dropna()

# Feature engineering
df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype(int)

df["Weight"] = df["Weight"].str.replace("kg", "", regex=False)
df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
df = df.dropna(subset=["Weight"])

df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")
df = df.dropna(subset=["Inches"])

def extract_storage(mem):
    total = 0
    parts = mem.split("+")
    for p in parts:
        nums = re.findall(r"\d+", p)
        if not nums:
            continue
        size = int(nums[0])
        if "TB" in p:
            total += size * 1024
        elif "GB" in p:
            total += size
    return total

df["StorageGB"] = df["Memory"].apply(extract_storage)
df = df.drop(columns=["Memory"])

X = df.drop("Price", axis=1)
y = df["Price"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    results[name] = {
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds))
    }

# Select best model
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]

final_model = Pipeline([
    ("prep", preprocessor),
    ("model", best_model)
])

final_model.fit(X, y)

# Save model and analysis
joblib.dump(final_model, "laptop_price_model.pkl")

with open("model_analysis.json", "w") as f:
    json.dump({
        "results": results,
        "best_model": best_model_name
    }, f, indent=4)

print("Model trained.")
print("Best Model:", best_model_name)
print("Analysis saved to model_analysis.json")
