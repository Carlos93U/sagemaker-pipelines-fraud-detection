"""SageMaker Pipeline definition (SDK V3): SplitData -> Train -> Evaluate -> Condition.

Only *builds* objects; it never calls AWS until the runner does upsert/start.
The container scripts live in code/ (preprocess.py, evaluate.py) and are the
single source of truth (they are not generated with %%writefile).
"""

import json
from typing import Any

from sagemaker.core import image_uris
from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (ProcessingInput, ProcessingS3Input,
                                   ProcessingOutput, ProcessingS3Output)
from sagemaker.core.workflow.functions import JsonGet, Join
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.core.model_metrics import MetricsSource, ModelMetrics
from sagemaker.core.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.fail_step import FailStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig

from . import config
from .session import Sessions

# Caching: same input + same code => reuses the previous run (<= 2h).
CACHE_CONFIG = CacheConfig(enable_caching=True, expire_after="PT2H")


def build_pipeline(sessions: "Sessions", input_data_s3: str | None = None) -> Pipeline:
    """Builds the full DAG and returns the Pipeline object (without running it)."""
    input_data_s3 = input_data_s3 or sessions.raw_data_s3

    sklearn_image = image_uris.retrieve(framework="sklearn", region=sessions.region,
                                        version="1.4-2", py_version="py3",
                                        instance_type=config.INSTANCE_TYPE)
    xgboost_image = image_uris.retrieve(framework="xgboost", region=sessions.region,
                                        version="1.7-1", py_version="py3",
                                        instance_type=config.INSTANCE_TYPE)

    # --- Pipeline parameters (configurable on EVERY run) ---
    input_data_param = ParameterString(name="InputData", default_value=input_data_s3)
    training_instance_type = ParameterString(name="TrainingInstanceType",
                                             default_value=config.INSTANCE_TYPE)
    metric_name = ParameterString(name="MetricName", default_value=config.METRIC_NAME)
    metric_threshold = ParameterFloat(name="MetricThreshold",
                                      default_value=config.METRIC_THRESHOLD)
    approval_status = ParameterString(name="ModelApprovalStatus",
                                      default_value=config.APPROVAL_STATUS)

    # ------------------------------------------------------------
    # SplitData (ProcessingStep, with caching)
    # ------------------------------------------------------------
    processor = ScriptProcessor(command=["python3"], image_uri=sklearn_image,
                                instance_type=training_instance_type, instance_count=1,
                                base_job_name="fraud-",
                                sagemaker_session=sessions.pipeline_session,
                                role=sessions.role)
    split_args = processor.run(
        inputs=[ProcessingInput(
            input_name="input-1",
            s3_input=ProcessingS3Input(s3_uri=input_data_param,
                                       local_path="/opt/ml/processing/input",
                                       s3_data_type="S3Prefix",
                                       s3_input_mode="File"))],
        outputs=[
            ProcessingOutput(output_name=train,
                             s3_output=ProcessingS3Output(
                                 s3_uri=f"s3://{sessions.bucket}/{sessions.prefix}/{name}",
                                 local_path=f"/opt/ml/processing/{train}",
                                 s3_upload_mode="EndOfJob"))
            for train, name in [("train", "data/train"), ("validation", "data/validation"),
                                ("test", "data/test"), ("baseline", "baseline")]
        ],
        code=config.code_path("preprocess.py"),
        arguments=["--input-file", config.RAW_FILE],
    )
    step_process = ProcessingStep(name="SplitData", step_args=split_args,
                                  cache_config=CACHE_CONFIG)

    # ------------------------------------------------------------
    # TrainFraudXGB (TrainingStep)
    # ------------------------------------------------------------
    model_trainer = ModelTrainer(
        training_image=xgboost_image,
        compute=Compute(instance_type=training_instance_type, instance_count=1,
                        volume_size_in_gb=config.VOLUME_SIZE_GB),
        base_job_name="fraud-xgb",
        output_data_config=OutputDataConfig(
            s3_output_path=f"s3://{sessions.bucket}/{sessions.prefix}/training-output"),
        sagemaker_session=sessions.pipeline_session,
        role=sessions.role,
        hyperparameters=config.XGB_HYPERPARAMS,
        input_data_config=[
            InputData(channel_name="train",
                      data_source=step_process.properties.ProcessingOutputConfig
                          .Outputs["train"].S3Output.S3Uri,
                      content_type="text/csv"),
            InputData(channel_name="validation",
                      data_source=step_process.properties.ProcessingOutputConfig
                          .Outputs["validation"].S3Output.S3Uri,
                      content_type="text/csv"),
        ],
    )
    train_args = model_trainer.train()
    step_train = TrainingStep(name="TrainFraudXGB", step_args=train_args)

    # ------------------------------------------------------------
    # EvaluateFraud (ProcessingStep, with caching)
    # ------------------------------------------------------------
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation",
                                     path="evaluation.json")
    eval_processor = ScriptProcessor(command=["python3"], image_uri=xgboost_image,
                                     instance_type=training_instance_type, instance_count=1,
                                     base_job_name="fraud-eval",
                                     sagemaker_session=sessions.pipeline_session,
                                     role=sessions.role)
    eval_args = eval_processor.run(
        inputs=[
            ProcessingInput(input_name="model",
                            s3_input=ProcessingS3Input(
                                s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                                local_path="/opt/ml/processing/model",
                                s3_data_type="S3Prefix", s3_input_mode="File")),
            ProcessingInput(input_name="valid",
                            s3_input=ProcessingS3Input(
                                s3_uri=step_process.properties.ProcessingOutputConfig
                                    .Outputs["validation"].S3Output.S3Uri,
                                local_path="/opt/ml/processing/valid",
                                s3_data_type="S3Prefix", s3_input_mode="File")),
        ],
        outputs=[ProcessingOutput(output_name="evaluation",
                                  s3_output=ProcessingS3Output(
                                      s3_uri=f"s3://{sessions.bucket}/{sessions.prefix}/evaluation",
                                      local_path="/opt/ml/processing/evaluation",
                                      s3_upload_mode="EndOfJob"))],
        code=config.code_path("evaluate.py"),
        arguments=["--metric", metric_name],      # MetricName controls WHICH metric decides
    )
    step_eval = ProcessingStep(name="EvaluateFraud", step_args=eval_args,
                               property_files=[evaluation_report],
                               cache_config=CACHE_CONFIG)

    # ------------------------------------------------------------
    # CheckFraudMetric (ConditionStep) + FraudThresholdNotMet (FailStep)
    # ------------------------------------------------------------
    metric_value = JsonGet(step_name=step_eval.name, property_file=evaluation_report,
                           json_path="binary_classification_metrics.metric.value")
    cond_has_metric = ConditionGreaterThanOrEqualTo(left=metric_value,
                                                    right=metric_threshold)
    step_fail = FailStep(
        name="FraudThresholdNotMet",
        error_message="The chosen metric did not reach the configured threshold: "
                      "no model is registered.",
    )

    # ------------------------------------------------------------
    # RegisterFraudModel (ModelStep)
    # ------------------------------------------------------------
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on="/", values=[
                step_eval.properties.ProcessingOutputConfig
                    .Outputs["evaluation"].S3Output.S3Uri,
                "evaluation.json",
            ]),
            content_type="application/json",
        )
    )
    model_builder = ModelBuilder(
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=xgboost_image,
        sagemaker_session=sessions.pipeline_session,
        role_arn=sessions.role,
    )
    step_register = ModelStep(
        name="RegisterFraudModel",
        step_args=model_builder.register(
            model_package_group_name=config.MODEL_PACKAGE_GROUP,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=[config.INSTANCE_TYPE],
            transform_instances=[config.INSTANCE_TYPE],
            model_metrics=model_metrics,           # metrics visible in the Registry
            approval_status=approval_status,       # PendingManualApproval by default
        ),
    )

    step_cond = ConditionStep(
        name="CheckFraudMetric",
        conditions=[cond_has_metric],
        if_steps=[step_register],      # PASS -> registers a version in the Model Registry
        else_steps=[step_fail],        # FAIL -> stops as Failed (and alerts via SNS)
    )

    return Pipeline(
        name=config.PIPELINE_NAME,
        parameters=[input_data_param, training_instance_type,
                    metric_name, metric_threshold, approval_status],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=sessions.pipeline_session,
    )


def preview_definition(pipeline: "Pipeline") -> dict[str, Any]:
    """Returns the JSON definition of the pipeline (step before upsert)."""
    return json.loads(pipeline.definition())
