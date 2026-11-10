"""
House Price Prediction using Linear Regression
Dataset: Indian House Price Prediction
Target: TARGET(PRICE_IN_LACS)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "train.csv"
MODEL_FILE = ROOT / "house_price_model.joblib"
PLOT_FILE = ROOT / "actual_vs_predicted.png"

TARGET = "TARGET(PRICE_IN_LACS)"


def main():

    # 1. Load dataset
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Place train.csv inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("---------------------------")

    print("Shape:", df.shape)
    print("\nDataset Information:")
    df.info()

    print("\nBasic Statistics:")
    print(df.describe(include="all"))

    # 2. Check target column
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # 3. Separate features and target
    y = pd.to_numeric(df[TARGET], errors="coerce")

    X = df.drop(
        columns=[TARGET, "Id"],
        errors="ignore"
    )

    # Remove rows where target is missing
    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # 4. Identify numeric and categorical columns
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("\nNumeric Features:", numeric_features)
    print("\nCategorical Features:", categorical_features)

    # 5. Handle missing numeric values
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 6. Handle categorical values
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 7. Combine preprocessing
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # 8. Create Linear Regression model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # 9. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 10. Train model
    print("\nTraining model...")

    model.fit(X_train, y_train)

    # 11. Make predictions
    predictions = model.predict(X_test)

    # 12. Evaluate model
    print("\nHOUSE PRICE PREDICTION RESULTS")
    print("--------------------------------")

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

    print(
        "Mean Absolute Error:",
        f"{mean_absolute_error(y_test, predictions):,.2f}"
    )

    # 13. Visualize actual vs predicted prices
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.65
    )

    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())

    plt.plot(
        [low, high],
        [low, high],
        "r--",
        label="Perfect Prediction"
    )

    plt.xlabel("Actual House Price (Lakhs)")
    plt.ylabel("Predicted House Price (Lakhs)")
    plt.title("Actual vs Predicted House Prices")

    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=150)
    plt.close()

    print("\nGraph saved:", PLOT_FILE)

    # 14. Save trained model
    joblib.dump(model, MODEL_FILE)

    print("Model saved:", MODEL_FILE)

    # 15. Display sample predictions
    preview = pd.DataFrame({
        "Actual Price": y_test.iloc[:5].to_numpy(),
        "Predicted Price": predictions[:5]
    })

    print("\nSAMPLE PREDICTIONS")
    print("------------------")
    print(preview.to_string(index=False))

    print("\nPROJECT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()