"""Pipeline execution: upsert, start, wait for results and compare with the baseline."""

import json
import time
from typing import TYPE_CHECKING

import pandas as pd

from . import config
from .session import Sessions

if TYPE_CHECKING:
    from sagemaker.mlops.workflow.pipeline import Pipeline


def upsert_and_start(sessions: "Sessions", pipeline: "Pipeline", params: dict) -> str:
    """Creates/updates the pipeline in SageMaker, launches it and returns the execution ARN."""
    pipeline.upsert(role_arn=sessions.role)
    execution = pipeline.start(parameters=params)
    arn = execution.describe()["PipelineExecutionArn"]
    print("Execution ARN:", arn)
    return arn


def wait_execution(sessions: "Sessions", arn: str,
                   timeout_min: int = 30, sleep: int = 20) -> str:
    """Waits for the execution to finish; returns its final status."""
    start = time.time()
    status = "Executing"
    while time.time() - start < timeout_min * 60:
        status = sessions.sm_client.describe_pipeline_execution(
            PipelineExecutionArn=arn)["PipelineExecutionStatus"]
        print(f"  {time.strftime('%H:%M:%S')} -> {status}")
        if status in ("Succeeded", "Failed", "Stopped"):
            break
        time.sleep(sleep)
    return status


def report_steps(sessions: "Sessions", arn: str) -> list[dict]:
    """Prints and returns the status of every step of the execution."""
    steps = sessions.sm_client.list_pipeline_execution_steps(
        PipelineExecutionArn=arn)["PipelineExecutionSteps"]
    for step in steps:
        print(f"  {step['StepName']:22s} -> {step['StepStatus']}")
    return steps


def read_s3_json(sessions: "Sessions", uri: str) -> dict:
    bucket, key = uri.replace("s3://", "").split("/", 1)
    body = sessions.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


def compare_with_baseline(sessions: "Sessions",
                          threshold: float = config.METRIC_THRESHOLD) -> pd.DataFrame:
    """Compares the main model (evaluation.json) with the baseline (baseline.json)."""
    eval_uri = f"s3://{sessions.bucket}/{sessions.prefix}/evaluation/evaluation.json"
    baseline_uri = f"s3://{sessions.bucket}/{sessions.prefix}/baseline/baseline.json"
    eval_metrics = read_s3_json(sessions, eval_uri)
    baseline_metrics = read_s3_json(sessions, baseline_uri)

    rows = [
        {
            "Metric": m,
            "Baseline (LogisticRegression)": round(baseline_metrics["baseline"][m], 4),
            "Main (XGBoost)": round(
                eval_metrics["binary_classification_metrics"][m]["value"], 4),
        }
        for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]

    chosen = eval_metrics["metric_used"]
    value = eval_metrics["binary_classification_metrics"][chosen]["value"]
    print(f"Chosen metric: {chosen} = {round(value, 4)}  (threshold: {threshold})")
    print("-> PASS" if value >= threshold
          else "-> FAIL (go to section 11 for the alert flow)")
    return pd.DataFrame(rows)
