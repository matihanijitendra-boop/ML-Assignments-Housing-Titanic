"""Titanic survival prediction using Logistic Regression.

Expected dataset: Kaggle Titanic competition train.csv
Expected target column: Survived
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "train.csv"
MODEL_FILE = ROOT / "titanic_survival_model.joblib"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Download Kaggle Titanic train.csv and put it in the data folder."
        )

    df = pd.read_csv(DATA_FILE)
    if "Survived" not in df.columns:
        raise ValueError("The CSV must contain the target column 'Survived'.")

    # Drop ID/name/ticket/cabin text fields; use passenger-related predictors.
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    y = df["Survived"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\\nTITANIC SURVIVAL PREDICTION – LOGISTIC REGRESSION")
    print("-------------------------------------------------")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")
    print(f"Accuracy     : {accuracy_score(y_test, predictions):.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_FILE)
    print(f"\\nModel saved: {MODEL_FILE}")

    print("\\nExample predictions (0 = Did not survive, 1 = Survived):")
    preview = pd.DataFrame({
        "Actual": y_test.iloc[:10].to_numpy(),
        "Predicted": predictions[:10],
    })
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
