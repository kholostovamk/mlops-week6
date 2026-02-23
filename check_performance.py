import json
import os
import sys


def main() -> None:
    threshold = 0.99  # minimum acceptable accuracy here

    eval_path = os.path.join("artifacts", "eval.json")
    if not os.path.exists(eval_path):
        print(f"Missing evaluation file: {eval_path}", file=sys.stderr)
        sys.exit(2)

    with open(eval_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    acc = float(metrics.get("accuracy", 0.0))
    print(f"THRESHOLD={threshold:.4f}")
    print(f"ACCURACY={acc:.4f}")

    if acc < threshold:
        print("FAIL: model underperforms threshold", file=sys.stderr)
        sys.exit(1)

    print("PASS: model meets threshold")
    sys.exit(0)


if __name__ == "__main__":
    main()