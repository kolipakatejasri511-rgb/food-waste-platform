import random
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

random.seed(42)

days = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

categories = [
    "Rice", "Meals", "Snacks", "Bread", "Vegetables"
]

rows = []

for _ in range(600):
    day = random.choice(days)
    category = random.choice(categories)
    customers = random.randint(30, 300)
    prepared = customers * random.uniform(0.18, 0.35)
    event = random.randint(0, 1)

    # Simulated data for a college prototype.
    demand_factor = random.uniform(0.65, 1.05)
    if event:
        demand_factor += 0.08
    if day in ["Saturday", "Sunday"]:
        demand_factor -= 0.08

    sold = min(prepared, customers * 0.25 * demand_factor)
    surplus = max(0, prepared - sold)

    rows.append({
        "day": day,
        "category": category,
        "customers": customers,
        "prepared_kg": round(prepared, 2),
        "event": event,
        "surplus_kg": round(surplus, 2)
    })

df = pd.DataFrame(rows)
df.to_csv("food_data.csv", index=False)

features = [
    "day", "category", "customers",
    "prepared_kg", "event"
]
target = "surplus_kg"

X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["day", "category"]
        )
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Absolute Error:",
      round(mean_absolute_error(y_test, predictions), 2), "kg")

print("R2 score:",
      round(r2_score(y_test, predictions), 3))

joblib.dump(model, "food_surplus_model.joblib")
print("Model saved successfully!")