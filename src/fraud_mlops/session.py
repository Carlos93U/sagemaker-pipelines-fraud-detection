"""AWS sessions and clients: one reusable call for every module.

* Session()         -> runs now (S3, deploy, queries, cleanup).
* PipelineSession() -> captures the pipeline intent; the service runs it on launch.
"""

from dataclasses import dataclass
from typing import Any

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.workflow.pipeline_context import PipelineSession

from . import config


@dataclass
class Sessions:
    """Groups the role, bucket and every client used by the package modules."""

    region: str
    role: str
    bucket: str
    prefix: str
    sagemaker_session: Session
    pipeline_session: PipelineSession
    sm_client: Any
    sm_runtime: Any
    s3: Any
    events: Any
    sns: Any
    iam: Any
    sts: Any

    @property
    def raw_key(self) -> str:
        return f"{self.prefix}/{config.DATA_PREFIX}/{config.RAW_FILE}"

    @property
    def raw_data_s3(self) -> str:
        return f"s3://{self.bucket}/{self.raw_key}"


def get_sessions(region: str | None = None, role_arn: str | None = None) -> Sessions:
    """Creates the sessions/clients. The role resolves as: argument > env var > auto-detect."""
    region = region or config.AWS_REGION
    boto3.setup_default_session(region_name=region)

    sagemaker_session = Session()
    role = role_arn or config.SAGEMAKER_ROLE_ARN
    if not role:
        try:
            role = get_execution_role()
            print("Role auto-detected:", role)
        except Exception as exc:
            raise RuntimeError(
                "Could not auto-detect the SageMaker role. "
                "Set the SAGEMAKER_ROLE_ARN environment variable with your ARN "
                "(e.g. arn:aws:iam::<account>:role/<role>) and run again."
            ) from exc

    return Sessions(
        region=region,
        role=role,
        bucket=sagemaker_session.default_bucket(),   # sagemaker-<region>-<account>
        prefix=config.RESOURCE_PREFIX,
        sagemaker_session=sagemaker_session,
        pipeline_session=PipelineSession(),
        sm_client=sagemaker_session.sagemaker_client,
        sm_runtime=boto3.client("sagemaker-runtime", region_name=region),
        s3=boto3.client("s3", region_name=region),
        events=boto3.client("events", region_name=region),
        sns=boto3.client("sns", region_name=region),
        iam=boto3.client("iam", region_name=region),
        sts=boto3.client("sts", region_name=region),
    )
