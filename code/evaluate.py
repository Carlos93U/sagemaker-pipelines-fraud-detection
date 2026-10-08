# code/evaluate.py  (runs in the SageMaker XGBoost container)
# Evaluates model.tar.gz on validation.csv and writes evaluation.json.
#   {"binary_classification_metrics": {<metric>: {value}, ...},
#    "binary_classification_metrics": {..., "metric": {"value": <chosen>}}}
import argparse
import json
import os
import pathlib
import tarfile

import numpy as np
import xgboost as xgb

ALL_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def roc_auc_np(y, score):
    order = np.argsort(score)
    y_s = y[order]
    n_pos = float((y_s == 1).sum())
    n_neg = float((y_s == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(len(y_s)) + 1
    sum_pos = float(ranks[y_s == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="roc_auc", choices=ALL_METRICS)
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--valid-file", type=str, default="/opt/ml/processing/valid/validation.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    with tarfile.open(os.path.join(args.model_dir, "model.tar.gz")) as tar:
        tar.extractall(path=args.model_dir)
    booster = xgb.Booster()
    booster.load_model(os.path.join(args.model_dir, "xgboost-model"))

    valid = np.loadtxt(args.valid_file, delimiter=",")
    X, y_true = valid[:, 1:], valid[:, 0].astype(int)
    y_score = booster.predict(xgb.DMatrix(X))
    y_pred = (y_score >= 0.5).astype(int)

    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())

    metrics = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1e-9),
        "roc_auc": roc_auc_np(y_true, y_score),
    }

    report = {
        "binary_classification_metrics": {
            name: {"value": float(metrics[name])} for name in ALL_METRICS
        },
        "metric_used": args.metric,
    }
    # FIXED key for the pipeline JsonGet (metric controlled by --metric):
    report["binary_classification_metrics"]["metric"] = {
        "value": report["binary_classification_metrics"][args.metric]["value"]
    }

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("EVALUATION:", json.dumps(report))


if __name__ == "__main__":
    main()
