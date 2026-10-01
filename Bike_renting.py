import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Load and prepare the data
# --------------------------------------------------

df = pd.read_csv("day.csv")

df["dteday"] = pd.to_datetime(df["dteday"])

# Sort chronologically because this is a forecasting problem
df = df.sort_values("dteday").reset_index(drop=True)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

X = df.drop(
    columns=[
        "instant",
        "dteday",
        "cnt",
        "casual",
        "registered"
    ]
)

y = df["cnt"]


# --------------------------------------------------
# 3. Chronological train/test split
# --------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# --------------------------------------------------
# 4. Feature preprocessing
# --------------------------------------------------

numerical_features = [
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

categorical_features = [
    "season",
    "yr",
    "mnth",
    "weekday",
    "workingday",
    "weathersit",
    "holiday"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 5. Create models
# --------------------------------------------------

models = {
    "Dummy Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="mean"))
    ]),

    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=40
            )
        )
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                random_state=40
            )
        )
    ])
}


# --------------------------------------------------
# 6. Time-series cross-validation
# --------------------------------------------------

tscv = TimeSeriesSplit(n_splits=5)


print("\nCross-validation results")
print("-" * 45)

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=tscv,
        scoring="neg_mean_absolute_error"
    )

    mean_mae = -scores.mean()

    cv_results[name] = mean_mae

    print(f"{name}:")
    print("MAE per fold:", -scores)
    print("Mean MAE:", mean_mae)
    print()


# --------------------------------------------------
# 7. Train models and evaluate on test data
# --------------------------------------------------

print("\nTest-set results")
print("-" * 45)

test_results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    test_results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)


# --------------------------------------------------
# 8. Plot Linear Regression predictions
# --------------------------------------------------

linear_model = models["Linear Regression"]

linear_predictions = linear_model.predict(X_test)

plt.figure(figsize=(12, 5))

plt.plot(
    df["dteday"].iloc[split_index:].values,
    y_test.values,
    label="Actual rentals"
)

plt.plot(
    df["dteday"].iloc[split_index:].values,
    linear_predictions,
    label="Linear Regression"
)

plt.xlabel("Date")
plt.ylabel("Number of rentals")
plt.title("Actual vs Predicted Bike Rentals")

plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
linear_model = models["Linear Regression"]
joblib.dump(linear_model, "bike_model.pkl")
print("Linear Regression model saved.")