"""Train the notebook-aligned mistriage model and save to `models/mistriage_kaggle_rf.joblib`."""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.kaggle_mistriage import train_and_save  # noqa: E402

if __name__ == "__main__":
    out = train_and_save(ROOT / "models" / "mistriage_kaggle_rf.joblib")
    print("Saved:", out["path"])
    print("Train accuracy:", round(out["train_accuracy"], 4))
    print("Test accuracy:", round(out["test_accuracy"], 4))
