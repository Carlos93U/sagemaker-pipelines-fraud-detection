"""Synthetic fraud dataset (Kaggle Credit-Card Fraud style) and its upload to S3."""

import io
from typing import TYPE_CHECKING

import pandas as pd

from . import config

if TYPE_CHECKING:
    from .session import Sessions


def make_fraud_dataframe(seed: int = 42, n_samples: int = 5000) -> pd.DataFrame:
    """Imbalanced binary dataset (~30% fraud) with 20 numeric features."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples, n_features=20,
        n_informative=15, n_redundant=2, n_repeated=0,
        n_classes=2, n_clusters_per_class=2,
        flip_y=0.03, weights=[0.7, 0.3],   # 70% not-fraud / 30% fraud
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=config.FEATURE_COLS)
    df[config.TARGET_COL] = y
    return df


def plot_overview(df: pd.DataFrame) -> None:
    """Lightweight visual summary: class frequency + 'amount' distribution."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    df[config.TARGET_COL].map({0: "Not Fraud", 1: "Fraud"}).value_counts().plot.bar(
        ax=axes[0], color=["#2b8a3e", "#c92a2a"])
    axes[0].set_title("Class frequency")
    df.boxplot(column="amount", by=config.TARGET_COL, ax=axes[1], grid=False)
    axes[1].set_title("amount by class")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()


def upload_dataframe(sessions: "Sessions", df: pd.DataFrame, key: str | None = None) -> str:
    """Uploads the DataFrame as CSV to the landing zone; returns the s3:// URI."""
    key = key or sessions.raw_key
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    sessions.s3.put_object(Bucket=sessions.bucket, Key=key, Body=buffer.getvalue())
    return f"s3://{sessions.bucket}/{key}"


def sample_probe(df: pd.DataFrame, n: int = 5, seed: int = 7) -> pd.DataFrame:
    """Sample rows that will be used in the Inference section."""
    return df.sample(n=n, random_state=seed)
