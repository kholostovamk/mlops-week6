import os
import subprocess


def test_training_script_runs():

    if os.path.exists("artifacts/model.pkl"):
        os.remove("artifacts/model.pkl")
    if os.path.exists("artifacts/metrics.json"):
        os.remove("artifacts/metrics.json")

    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"
    assert False, "Intentional unit test failure for CI demonstration"


def test_evaluation_script_runs():
    result = subprocess.run(
        ["python", "evaluate.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Evaluation failed:\n{result.stderr}"
    assert os.path.exists("artifacts/eval.json"), "eval.json was not created"