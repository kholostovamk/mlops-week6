import json
import os
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
    model_path = os.path.join("artifacts", "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model artifact: {model_path}. Run train.py first.")

    model = joblib.load(model_path)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("artifacts", exist_ok=True)
    eval_path = os.path.join("artifacts", "eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(acc)}, f)

    print(f"EVAL_SAVED={eval_path}")
    print(f"EVAL_ACCURACY={acc:.4f}")


if __name__ == "__main__":
    main()