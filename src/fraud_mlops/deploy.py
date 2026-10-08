"""Model Registry: group, approval gate and endpoint idempotency."""

from typing import Any

from botocore.exceptions import ClientError

from sagemaker.core.resources import ModelPackage
from sagemaker.serve.model_builder import ModelBuilder

from . import config
from .session import Sessions


def ensure_model_package_group(sessions: "Sessions") -> None:
    """Creates the Model Registry group if it does not exist (idempotent)."""
    try:
        sessions.sm_client.describe_model_package_group(
            ModelPackageGroupName=config.MODEL_PACKAGE_GROUP)
        print("Group exists:", config.MODEL_PACKAGE_GROUP)
    except ClientError:
        sessions.sm_client.create_model_package_group(
            ModelPackageGroupName=config.MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="Fraud Detection MLOps - portfolio.",
        )
        print("Group created:", config.MODEL_PACKAGE_GROUP)


def latest_model_version(sessions: "Sessions") -> dict[str, Any]:
    """Returns the summary of the newest registered version (newest last)."""
    versions = sessions.sm_client.list_model_packages(
        ModelPackageGroupName=config.MODEL_PACKAGE_GROUP,
        MaxResults=5)["ModelPackageSummaryList"]
    for version in versions:
        print({"version": version["ModelPackageVersion"],
               "status": version.get("ModelPackageStatus"),
               "approval": version.get("ModelApprovalStatus")})

    if not versions:
        raise RuntimeError(
            "No versions registered: the Conditional did not register a model (FAIL?).")
    return versions[0]


def ensure_approved(sessions: "Sessions", package_arn: str,
                    description: str = "Approved manually from the notebook (demo).") -> str:
    """Approval gate: records the human approval if the package is not Approved yet."""
    detail = sessions.sm_client.describe_model_package(ModelPackageName=package_arn)
    approval = detail.get("ModelApprovalStatus")
    print("Approval status:", approval)

    if approval != "Approved":
        print("GATE: no deployment without approval. Allowing the approval here "
              "(simulates the human/config):")
        sessions.sm_client.update_model_package(
            ModelPackageArn=package_arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription=description,
        )
        print("Version approved: OK")
        return "Approved"
    return approval


def ensure_endpoint(sessions: "Sessions", package_arn: str) -> tuple[Any, bool]:
    """Deploys the endpoint if it does not exist (idempotent). Returns (endpoint, created?)."""
    try:
        status = sessions.sm_client.describe_endpoint(
            EndpointName=config.ENDPOINT_NAME)["EndpointStatus"]
        print("Endpoint already exists:", status)
        return None, False
    except sessions.sm_client.exceptions.ClientError:
        pass

    package = ModelPackage.get(model_package_name=package_arn)
    container = package.inference_specification.containers[0]

    builder = ModelBuilder(
        s3_model_data_url=container.model_data_url,
        image_uri=container.image,
        role_arn=sessions.role,
        sagemaker_session=sessions.sagemaker_session,
    )
    builder.build(model_name=config.MODEL_NAME)
    endpoint = builder.deploy(
        instance_type=config.INSTANCE_TYPE,
        initial_instance_count=1,
        endpoint_name=config.ENDPOINT_NAME,
    )
    print("Endpoint created/in service:", config.ENDPOINT_NAME)

    status = sessions.sm_client.describe_endpoint(
        EndpointName=config.ENDPOINT_NAME)["EndpointStatus"]
    print("Endpoint status:", status)
    return endpoint, True
