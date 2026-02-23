import os
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main() -> None:
    # Deterministic, small dataset (built-in)
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    metrics_path = os.path.join("artifacts", "metrics.json")

    joblib.dump(model, model_path)    # Save model artifact

    metrics = {"accuracy": float(acc)}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    # Parseable CI logs
    print(f"MODEL_SAVED={model_path}")
    print(f"METRICS_SAVED={metrics_path}")
    print(f"ACCURACY={acc:.4f}")


if __name__ == "__main__":
    main()