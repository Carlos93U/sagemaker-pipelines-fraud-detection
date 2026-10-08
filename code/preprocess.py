# code/preprocess.py  (runs in the SageMaker Scikit-learn container)
# Fraud: train/validation/test split + baseline (LogisticRegression).
import argparse
import json
import os
import pathlib

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split


def compute_metrics(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="raw.csv")
    parser.add_argument("--baseline-dir", type=str, default="/opt/ml/processing/baseline")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join("/opt/ml/processing/input", args.input_file))
    train, other = train_test_split(df, test_size=0.30, random_state=42, stratify=df["target"])
    validation, test = train_test_split(other, test_size=1 / 3, random_state=42,
                                        stratify=other["target"])

    # formats XGBoost consumes: target FIRST and no header
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        out_dir = f"/opt/ml/processing/{name}"
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        part = part[["target"] + [c for c in part.columns if c != "target"]]
        part.to_csv(os.path.join(out_dir, f"{name}.csv"), header=False, index=False)
        print(f"{name}: {len(part)} rows")

    # --- Baseline: logistic regression on validation ---
    feat = [c for c in df.columns if c != "target"]
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train[feat].values, train["target"].values)
    y_score = lr.predict_proba(validation[feat].values)[:, 1]
    report = {"baseline": compute_metrics(validation["target"].values, y_score)}

    pathlib.Path(args.baseline_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.baseline_dir, "baseline.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("BASELINE:", json.dumps(report))


if __name__ == "__main__":
    main()
