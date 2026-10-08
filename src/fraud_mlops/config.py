"""Constants of the fraud-mlops project.

All values are "defaults": most of them can be overridden on each pipeline run
(InputData, MetricName, MetricThreshold, ModelApprovalStatus parameters). The
optional environment variables let you adapt the project to another
account/region without editing code.
"""

import os
from pathlib import Path

# ------------------------------------------------------------
# SageMaker resources
# ------------------------------------------------------------
PIPELINE_NAME = "fraud-detection-mlops"                # SageMaker Pipeline
MODEL_PACKAGE_GROUP = "fraud-detection-mlops-mpg"      # stable group in Model Registry
MODEL_NAME = "fraud-detection-xgboost"                 # Model (for the endpoint)
ENDPOINT_NAME = "fraud-detection-endpoint"             # real-time endpoint

# ------------------------------------------------------------
# S3
# ------------------------------------------------------------
RESOURCE_PREFIX = "fraud-mlops"                        # general prefix in S3
DATA_PREFIX = "ingest"                                 # data 'landing zone'
RAW_FILE = "raw.csv"                                   # input file name

# ------------------------------------------------------------
# Instances (small: lab / controlled cost)
# ------------------------------------------------------------
INSTANCE_TYPE = "ml.m5.large"                          # processing / training / endpoint
VOLUME_SIZE_GB = 30

# ------------------------------------------------------------
# Model + evaluation criteria
# ------------------------------------------------------------
FEATURE_COLS = ["time", "amount"] + [f"v{i:02d}" for i in range(1, 19)]   # 20 features
TARGET_COL = "target"                                  # 1 = Fraud, 0 = Not Fraud
XGB_HYPERPARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "num_round": 100,
    "max_depth": 6,
    "eta": 0.2,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# Default criterion for the Conditional (changeable at run time)
METRIC_NAME = "roc_auc"           # accuracy | precision | recall | f1 | roc_auc
METRIC_THRESHOLD = 0.70           # if metric >= threshold -> PASS

# Model Registry registration
APPROVAL_STATUS = "PendingManualApproval"              # requires human approval

# ------------------------------------------------------------
# Events / alerts
# ------------------------------------------------------------
EVENT_RULE_TRIGGER = "fraud-new-data-trigger"          # S3 -> EventBridge -> Pipeline
EVENT_RULE_FAIL = "fraud-pipeline-failed-alert"        # Pipeline Failed -> SNS
EVENT_RULE_START = "fraud-pipeline-started-alert"      # Pipeline Executing (start) -> SNS
SNS_TOPIC_NAME = "fraud-mlops-alerts"                  # notifications topic
EVENTS_ROLE_NAME = "fraud-mlops-events-role"           # EventBridge role
EVENTS_ROLE_POLICY = "start-fraud-pipeline"            # inline policy of the role
TRIGGER_TARGET_ID = "start-fraud-pipeline"             # target id (rule 1)
FAIL_TARGET_ID = "notify-fail"                         # target id (rule 2)
START_TARGET_ID = "notify-start"                       # target id (rule 3)

# ------------------------------------------------------------
# Optional environment variables (zero personal data in the repo)
# ------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN")   # fallback if not auto-detected
SNS_ALERT_EMAIL = os.environ.get("SNS_ALERT_EMAIL")         # if unset, nobody subscribes

# ------------------------------------------------------------
# Local paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]     # repo root
CODE_DIR = PROJECT_ROOT / "code"                       # scripts that run in containers


def code_path(name: str) -> str:
    """Absolute path of a code/ script (for processor.run(code=...))."""
    path = CODE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Container script not found: {path}")
    return str(path)


def build_run_params(input_data_s3: str, **overrides) -> dict:
    """Default parameters for a pipeline run (+ overrides)."""
    params = {
        "InputData": input_data_s3,
        "TrainingInstanceType": INSTANCE_TYPE,
        "MetricName": METRIC_NAME,
        "MetricThreshold": METRIC_THRESHOLD,
        "ModelApprovalStatus": APPROVAL_STATUS,
    }
    params.update(overrides)
    return params
