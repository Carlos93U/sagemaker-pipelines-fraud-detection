"""Inference against the real-time endpoint (text/csv, one probability per row)."""

from typing import Any

import pandas as pd

from . import config
from .session import Sessions

"""
def _parse_probabilities(raw: bytes | str) -> list[float]:
    text = raw.decode() if isinstance(raw, bytes) else raw
    return [float(x) for x in text.strip().split("\n") if x != ""]
"""

def _parse_probabilities(raw: bytes | str) -> list[float]:
    if hasattr(raw, "body"):
        raw = raw.body

    if hasattr(raw, "read"):
        raw = raw.read()

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    return [float(x.strip()) for x in text.strip().splitlines() if x.strip()]

def invoke_payload(sessions: "Sessions", payload: str,
                   endpoint: Any = None) -> list[float]:
    """Invokes the endpoint with a CSV payload (rows separated by \\n)."""
    if endpoint is not None:
        out = endpoint.invoke(body=payload, content_type="text/csv", accept="text/csv")
        return _parse_probabilities(out)

    response = sessions.sm_runtime.invoke_endpoint(
        EndpointName=config.ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="text/csv",
        Body=payload,
    )
    return _parse_probabilities(response["Body"].read())


def predict_dataframe(sessions: "Sessions", df: pd.DataFrame,
                      endpoint: Any = None) -> pd.DataFrame:
    """Predicts for every row of df (without the target column)."""
    payload = "\n".join(
        ",".join(f"{v:.6f}" for v in row[config.FEATURE_COLS])
        for _, row in df.iterrows()
    )
    probs = invoke_payload(sessions, payload, endpoint=endpoint)

    result = df.copy()
    result["probability"] = probs
    result["prediction"] = (result["probability"] >= 0.5).astype(int)
    return result


def predict_row(sessions: "Sessions", row: pd.Series,
                endpoint: Any = None) -> float:
    """Predicts a single row; returns the fraud probability."""
    payload = ",".join(f"{v:.6f}" for v in row[config.FEATURE_COLS])
    return invoke_payload(sessions, payload, endpoint=endpoint)[0]
