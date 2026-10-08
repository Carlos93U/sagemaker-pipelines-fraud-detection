"""Idempotent cleanup: deletes the resources created by the lab."""

import time
from typing import TYPE_CHECKING, Any

from . import config
from .events import disable_s3_eventbridge
from .session import Sessions

if TYPE_CHECKING:
    from .pipeline import Pipeline


def cleanup_resources(sessions: "Sessions", pipeline: "Pipeline | None" = None,
                      topic_arn: str | None = None) -> None:
    """Deletes endpoint/model, pipeline, Model Registry groups, EventBridge rules,
    the S3->EventBridge notification, the events role and the SNS topic."""
    print("== Cleaning up endpoint/model ==")
    try:
        sessions.sm_client.delete_endpoint(EndpointName=config.ENDPOINT_NAME)
        print("- Endpoint deleted")
    except Exception as exc:
        print("- Endpoint:", exc)
    try:
        sessions.sm_client.delete_endpoint_config(EndpointConfigName=config.ENDPOINT_NAME)
        print("- EndpointConfig deleted")
    except Exception as exc:
        print("- EndpointConfig:", exc)
    try:
        sessions.sm_client.delete_model(ModelName=config.MODEL_NAME)
        print("- Model deleted")
    except Exception as exc:
        print("- Model:", exc)

    if pipeline is not None:
        print("== Cleaning up pipeline ==")
        try:
            pipeline.delete()
            print("- Pipeline deleted")
        except Exception as exc:
            if "NotFound" in type(exc).__name__ or "Could not find" in str(exc):
                print("- Pipeline: no longer exists")
            else:
                print("- Pipeline:", exc)

    cleanup_model_registry(sessions)

    print("== Cleaning up EventBridge / SNS / IAM ==")
    # getattr: if the kernel has an old config.py, warn instead of crashing.
    rule_pairs = [("EVENT_RULE_TRIGGER", "TRIGGER_TARGET_ID"),
                  ("EVENT_RULE_FAIL", "FAIL_TARGET_ID"),
                  ("EVENT_RULE_START", "START_TARGET_ID")]
    for name, target in rule_pairs:
        rule, target_id = getattr(config, name, None), getattr(config, target, None)
        if not rule:
            print(f"- WARNING: config.{name} does not exist (re-run the configuration "
                  f"cell or restart the kernel); rule NOT deleted")
            continue
        try:
            sessions.events.remove_targets(Rule=rule, Ids=[target_id])
        except Exception as exc:
            print("- targets", rule, ":", exc)
        try:
            sessions.events.delete_rule(Name=rule)
            print("- Rule deleted:", rule)
        except Exception as exc:
            print("- Rule", rule, ":", exc)

    try:
        disable_s3_eventbridge(sessions)
    except Exception as exc:
        print("- S3 notifications:", exc)

    try:
        sessions.iam.delete_role_policy(RoleName=config.EVENTS_ROLE_NAME,
                                        PolicyName=config.EVENTS_ROLE_POLICY)
    except Exception as exc:
        print("- RolePolicy:", exc)
    try:
        sessions.iam.delete_role(RoleName=config.EVENTS_ROLE_NAME)
        print("- Role deleted:", config.EVENTS_ROLE_NAME)
    except Exception as exc:
        print("- Role:", exc)

    if topic_arn:
        try:
            sessions.sns.delete_topic(TopicArn=topic_arn)
            print("- SNS topic deleted")
        except Exception as exc:
            print("- SNS:", exc)

    print("Cleanup completed (S3 data is kept; "
          "delete the prefix with cleanup_s3(delete=True) if you want).")


def cleanup_model_registry(sessions: "Sessions") -> None:
    """Deletes the project's Model Registry groups (versions included)."""
    print("== Cleaning up Model Registry ==")
    paginator = sessions.sm_client.get_paginator("list_model_package_groups")
    found = []
    for page in paginator.paginate(NameContains="fraud-detection-mlops-mpg"):
        found.extend(page.get("ModelPackageGroupSummaryList", []))
    if not found:
        print("- No project groups")
        return
    pkg_paginator = sessions.sm_client.get_paginator("list_model_packages")
    for group in found:
        name = group["ModelPackageGroupName"]
        try:
            for page in pkg_paginator.paginate(ModelPackageGroupName=name):
                for pkg in page.get("ModelPackageSummaryList", []):
                    # delete_model_package accepts a name or an ARN; summaries only carry the ARN
                    sessions.sm_client.delete_model_package(
                        ModelPackageName=pkg.get("ModelPackageArn")
                        or pkg["ModelPackageName"])
            # package deletion is asynchronous: retry deleting the group
            for attempt in range(4):
                try:
                    sessions.sm_client.delete_model_package_group(
                        ModelPackageGroupName=name)
                    print("- Group deleted:", name)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
        except Exception as exc:
            print("- Group", name, ":", exc)


def cleanup_s3(sessions: "Sessions", delete: bool = False) -> None:
    """Deletes (optional) every object under the project prefix in S3."""
    if not delete:
        print(f"S3 untouched (delete=False); would delete s3://{sessions.bucket}/"
              f"{sessions.prefix}/ if you enable it.")
        return

    paginator = sessions.s3.get_paginator("list_objects_v2")
    keys: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=sessions.bucket, Prefix=f"{sessions.prefix}/"):
        keys.extend({"Key": obj["Key"]} for obj in page.get("Contents", []))
    if keys:
        sessions.s3.delete_objects(Bucket=sessions.bucket, Delete={"Objects": keys})
        print(f"Deleted {len(keys)} objects from s3://{sessions.bucket}/{sessions.prefix}/")
    else:
        print("No objects to delete.")
