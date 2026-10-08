"""Event and alert components: S3 -> EventBridge -> Pipeline, and Pipeline
start/failure -> SNS (email).

Everything is idempotent: creating something that already exists is not an error.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import boto3

from . import config
from .session import Sessions

if TYPE_CHECKING:
    from .pipeline import Pipeline


def pipeline_arn(sessions: "Sessions") -> str:
    """ARN of the pipeline in the active account/region."""
    account_id = sessions.sts.get_caller_identity()["Account"]
    return (f"arn:aws:sagemaker:{sessions.region}:{account_id}:"
            f"pipeline/{config.PIPELINE_NAME}")


def enable_s3_eventbridge(sessions: "Sessions") -> None:
    """Enables S3 -> EventBridge notifications on the bucket (one time only)."""
    sessions.s3.put_bucket_notification_configuration(
        Bucket=sessions.bucket, NotificationConfiguration={"EventBridgeConfiguration": {}})
    print(f"EventBridge notifications enabled for s3://{sessions.bucket}")

    notification = sessions.s3.get_bucket_notification_configuration(Bucket=sessions.bucket)
    assert "EventBridgeConfiguration" in notification, "S3 -> EventBridge is not enabled"
    print("S3 -> EventBridge enabled")


def disable_s3_eventbridge(sessions: "Sessions") -> None:
    """Removes the S3 -> EventBridge notification from the bucket if it is active.

    Keeps any other notification (Topic/Queue/Lambda) the bucket already had.
    """
    notification = sessions.s3.get_bucket_notification_configuration(Bucket=sessions.bucket)
    if "EventBridgeConfiguration" not in notification:
        print("- S3 -> EventBridge was already disabled")
        return
    rest = {k: v for k, v in notification.items()
            if k in ("TopicConfigurations", "QueueConfigurations", "LambdaConfigurations")
            and v}
    sessions.s3.put_bucket_notification_configuration(
        Bucket=sessions.bucket, NotificationConfiguration=rest)
    print("- S3 -> EventBridge disabled on s3://" + sessions.bucket)


def ensure_events_role(sessions: "Sessions") -> str:
    """Role EventBridge uses to launch the pipeline; returns its ARN."""
    events_arn = pipeline_arn(sessions)
    print("Pipeline ARN:", events_arn)

    try:
        role = sessions.iam.create_role(
            RoleName=config.EVENTS_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "events.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }],
            }),
        )["Role"]
        print("Role created:", config.EVENTS_ROLE_NAME)
    except sessions.iam.exceptions.EntityAlreadyExistsException:
        role = sessions.iam.get_role(RoleName=config.EVENTS_ROLE_NAME)["Role"]
        print("Role already existed:", config.EVENTS_ROLE_NAME)

    policy_doc = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "sagemaker:StartPipelineExecution",
            "Resource": events_arn,
        }],
    })
    sessions.iam.put_role_policy(RoleName=config.EVENTS_ROLE_NAME,
                                 PolicyName=config.EVENTS_ROLE_POLICY,
                                 PolicyDocument=policy_doc)
    print("Policy attached to the events role.")

    return role["Arn"]


def ensure_eventbridge_publish(sessions: "Sessions", topic_arn: str) -> None:
    """Makes sure the SNS topic accepts Publish from events.amazonaws.com.

    Without this policy EventBridge cannot publish to the topic (the target was
    created through the API, and unlike the AWS console it does not add the permission).
    """
    policy = json.loads(sessions.sns.get_topic_attributes(TopicArn=topic_arn)
                        ["Attributes"].get("Policy", "{}"))
    statements = policy.setdefault("Statement", [])

    for statement in statements:
        principal = statement.get("Principal", {})
        actions = statement.get("Action", [])
        if (principal == {"Service": "events.amazonaws.com"}
                and "sns:Publish" in ([actions] if isinstance(actions, str) else actions)
                and topic_arn in statement.get("Resource", "")):
            print("SNS policy already allows publishing from EventBridge.")
            return

    statements.append({
        "Sid": "Allow_Publish_Events",
        "Effect": "Allow",
        "Principal": {"Service": "events.amazonaws.com"},
        "Action": "sns:Publish",
        "Resource": topic_arn,
    })
    sessions.sns.set_topic_attributes(TopicArn=topic_arn, AttributeName="Policy",
                                      AttributeValue=json.dumps(policy))
    print("SNS policy updated: events.amazonaws.com -> sns:Publish allowed.")


def ensure_sns_topic(sessions: "Sessions") -> str:
    """SNS alert topic; if SNS_ALERT_EMAIL is set, that email is subscribed."""
    topic_arn = sessions.sns.create_topic(Name=config.SNS_TOPIC_NAME)["TopicArn"]
    print("SNS topic:", topic_arn)

    ensure_eventbridge_publish(sessions, topic_arn)

    if config.SNS_ALERT_EMAIL:
        sessions.sns.subscribe(TopicArn=topic_arn, Protocol="email",
                               Endpoint=config.SNS_ALERT_EMAIL)
        subscriptions = sessions.sns.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]
        state = next((s.get("SubscriptionArn", "unknown") for s in subscriptions
                      if s.get("Endpoint") == config.SNS_ALERT_EMAIL), "not found")
        if state == "PendingConfirmation":
            print(f"AVISO: the subscription of {config.SNS_ALERT_EMAIL} is PENDING "
                  "confirmation -> no emails will arrive until you confirm it.")
        else:
            print("Subscription confirmed for alerts:", config.SNS_ALERT_EMAIL)
    else:
        print("SNS_ALERT_EMAIL not set -> no email subscription "
              "(alerts are published anyway).")
    return topic_arn


def _pipeline_status_pattern(sessions: "Sessions", statuses: list[str]) -> str:
    """EventBridge pattern for execution status changes of this pipeline.

    The real event uses detail.currentPipelineExecutionStatus and detail.pipelineArn
    (NOT PipelineExecutionStatus/PipelineName, which do not exist in the payload).
    """
    return json.dumps({
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
        "detail": {
            "currentPipelineExecutionStatus": statuses,
            "pipelineArn": [{"prefix": pipeline_arn(sessions)}],
        },
    })


def ensure_event_rules(sessions: "Sessions", events_role_arn: str, topic_arn: str,
                       input_data_s3: str | None = None) -> None:
    """Rule 1: ObjectCreated on ingest/ -> launches the pipeline.

    Rule 2: Failed run -> SNS. Rule 3: Executing run (start) -> SNS.
    """
    input_data_s3 = input_data_s3 or sessions.raw_data_s3

    # --- Rule 1: new data -> start pipeline ---
    sessions.events.put_rule(
        Name=config.EVENT_RULE_TRIGGER,
        EventPattern=json.dumps({
            "source": ["aws.s3"],
            "detail-type": ["Object Created"],
            "detail": {
                "bucket": {"name": [sessions.bucket]},
                "object": {"key": [{"prefix": f"{sessions.prefix}/{config.DATA_PREFIX}/"}]},
            },
        }),
        State="ENABLED",
        Description="Launches the fraud pipeline when a new CSV arrives at ingest/",
    )
    sessions.events.put_targets(
        Rule=config.EVENT_RULE_TRIGGER,
        Targets=[{
            "Arn": pipeline_arn(sessions),
            "Id": config.TRIGGER_TARGET_ID,
            "RoleArn": events_role_arn,
            "SageMakerPipelineParameters": {
                "PipelineParameterList": [
                    {"Name": "InputData", "Value": input_data_s3},
                    {"Name": "MetricName", "Value": config.METRIC_NAME},
                    {"Name": "MetricThreshold", "Value": str(config.METRIC_THRESHOLD)},
                ]
            },
        }],
    )
    print("Rule in place:", config.EVENT_RULE_TRIGGER, "->", pipeline_arn(sessions))

    # --- Rule 2: pipeline Failed -> SNS notification (alert) ---
    sessions.events.put_rule(
        Name=config.EVENT_RULE_FAIL,
        EventPattern=_pipeline_status_pattern(sessions, ["Failed"]),
        State="ENABLED",
        Description="Alerts when a fraud pipeline execution fails",
    )
    sessions.events.put_targets(
        Rule=config.EVENT_RULE_FAIL,
        Targets=[{"Arn": topic_arn, "Id": config.FAIL_TARGET_ID}])
    print("Rule in place:", config.EVENT_RULE_FAIL, "->", topic_arn)

    # --- Rule 3: pipeline Executing (start) -> SNS notification ---
    sessions.events.put_rule(
        Name=config.EVENT_RULE_START,
        EventPattern=_pipeline_status_pattern(sessions, ["Executing"]),
        State="ENABLED",
        Description="Alerts when a fraud pipeline execution starts",
    )
    sessions.events.put_targets(
        Rule=config.EVENT_RULE_START,
        Targets=[{"Arn": topic_arn, "Id": config.START_TARGET_ID}])
    print("Rule in place:", config.EVENT_RULE_START, "->", topic_arn)


def wait_for_automatic_execution(sessions: "Sessions", tries: int = 20,
                                 sleep: int = 15) -> dict[str, Any] | None:
    """Waits for EventBridge to launch a pipeline run; returns its summary."""
    found = None
    for _ in range(tries):
        execs = sessions.sm_client.list_pipeline_executions(
            PipelineName=config.PIPELINE_NAME, MaxResults=5)
        latest = execs["PipelineExecutionSummaries"][0]
        if latest["PipelineExecutionStatus"] == "Executing":
            found = latest
            break
        time.sleep(sleep)

    if found is not None:
        print("Triggered automatically! New execution:", found["PipelineExecutionArn"])
    else:
        print("The automatic execution takes a while to show up.")
        execs = sessions.sm_client.list_pipeline_executions(
            PipelineName=config.PIPELINE_NAME, MaxResults=3)
        for execution in execs["PipelineExecutionSummaries"]:
            print("  -", execution["PipelineExecutionArn"],
                  execution["PipelineExecutionStatus"],
                  execution.get("PipelineExecutionTime", ""))
    return found


def publish_alert(sessions: "Sessions", topic_arn: str,
                  subject: str, message: str) -> None:
    """Publishes a test notification to the SNS topic."""
    sessions.sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
    print("SNS message published.")


def sample_execution_event(sessions: "Sessions", status: str,
                           previous: str | None = None) -> dict[str, Any]:
    """Sample event with the shape documented by SageMaker for EventBridge."""
    arn = pipeline_arn(sessions)
    detail: dict[str, Any] = {
        "currentPipelineExecutionStatus": status,
        "pipelineArn": arn,
        "pipelineExecutionArn": f"{arn}/execution/sample0000000",
        "executionStartTime": datetime.now(timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "pipelineExecutionDisplayName": "sample",
    }
    if previous:
        detail["previousPipelineExecutionStatus"] = previous
    return {
        "version": "0",
        "id": "00000000-0000-0000-0000-000000000000",
        "detail-type": "SageMaker Model Building Pipeline Execution Status Change",
        "source": "aws.sagemaker",
        "account": sessions.sts.get_caller_identity()["Account"],
        "time": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "region": sessions.region,
        "resources": [arn, detail["pipelineExecutionArn"]],
        "detail": detail,
    }


def check_event_patterns(sessions: "Sessions") -> dict[str, bool]:
    """Checks (without invoking targets) that the patterns match the real event.

    Uses events.test_event_pattern, the same API the console uses: it generates
    no cost and sends no notifications.
    """
    cases = [
        (config.EVENT_RULE_FAIL, sample_execution_event(sessions, "Failed", "Executing"), True),
        (config.EVENT_RULE_FAIL, sample_execution_event(sessions, "Succeeded", "Executing"), False),
        (config.EVENT_RULE_START, sample_execution_event(sessions, "Executing"), True),
        (config.EVENT_RULE_START, sample_execution_event(sessions, "Failed", "Executing"), False),
    ]
    results: dict[str, bool] = {}
    for rule_name, event, expected in cases:
        pattern = sessions.events.describe_rule(Name=rule_name)["EventPattern"]
        matched = sessions.events.test_event_pattern(
            EventPattern=pattern, Event=json.dumps(event))["Result"]
        status = event["detail"]["currentPipelineExecutionStatus"]
        ok = matched == expected
        results[f"{rule_name}:{status}"] = ok
        print(("OK  " if ok else "FAIL")
              + f" pattern {rule_name} with status {status}: "
              + ("matches" if matched else "does not match")
              + ("" if ok else " (unexpected)"))
    return results


def _metric_sum(cloudwatch: Any, namespace: str, metric: str,
                dimensions: list[dict], hours: int = 24) -> float:
    """Sum of a metric over the last `hours` hours (0 if there is no data)."""
    end = datetime.now(timezone.utc)
    datapoints = cloudwatch.get_metric_statistics(
        Namespace=namespace, MetricName=metric, Dimensions=dimensions,
        StartTime=end - timedelta(hours=hours), EndTime=end,
        Period=3600, Statistics=["Sum"])["Datapoints"]
    return round(sum(dp["Sum"] for dp in datapoints), 2)


def diagnose_alerts(sessions: "Sessions", topic_arn: str,
                    hours: int = 24) -> dict[str, Any]:
    """Diagnostics for the EventBridge -> SNS -> email chain (read only)."""
    report: dict[str, Any] = {"topic_arn": topic_arn, "rules": {}}

    # 1. Topic policy: can EventBridge publish?
    policy = json.loads(sessions.sns.get_topic_attributes(TopicArn=topic_arn)
                        ["Attributes"].get("Policy", "{}"))
    can_publish = any(
        stmt.get("Principal") == {"Service": "events.amazonaws.com"}
        and "sns:Publish" in (stmt.get("Action")
                              if isinstance(stmt.get("Action"), list) else [stmt.get("Action")])
        for stmt in policy.get("Statement", []))
    report["eventbridge_can_publish"] = can_publish
    print(("OK  " if can_publish else "FAIL")
          + " SNS policy: EventBridge -> sns:Publish "
          + ("allowed." if can_publish else "MISSING (nothing will arrive)."))

    # 2. Subscriptions: is any confirmed?
    subscriptions = sessions.sns.list_subscriptions_by_topic(TopicArn=topic_arn)["Subscriptions"]
    confirmed = [s for s in subscriptions
                 if str(s.get("SubscriptionArn", "")).startswith("arn:")]
    report["subscriptions"] = [
        {"endpoint": s.get("Endpoint"), "protocol": s.get("Protocol"),
         "confirmed": str(s.get("SubscriptionArn", "")).startswith("arn:")}
        for s in subscriptions]
    if not subscriptions:
        print("FAIL subscriptions: the topic has none.")
    for sub in report["subscriptions"]:
        print(("OK  " if sub["confirmed"] else "FAIL")
              + f" subscription {sub['protocol']}={sub['endpoint']}: "
              + ("confirmed." if sub["confirmed"] else "PENDING confirmation."))
    report["has_confirmed_subscription"] = bool(confirmed)

    # 3. Rules: pattern + targets
    for rule_name in (config.EVENT_RULE_TRIGGER, config.EVENT_RULE_FAIL,
                      config.EVENT_RULE_START):
        try:
            rule = sessions.events.describe_rule(Name=rule_name)
            targets = sessions.events.list_targets_by_rule(Rule=rule_name)["Targets"]
        except sessions.events.exceptions.ResourceNotFoundException:
            print(f"FAIL rule {rule_name}: DOES NOT EXIST.")
            report["rules"][rule_name] = {"exists": False}
            continue
        report["rules"][rule_name] = {
            "exists": True, "state": rule.get("State"),
            "pattern": rule.get("EventPattern"),
            "targets": [t.get("Arn") for t in targets],
        }
        print(f"{'OK  ' if rule.get('State') == 'ENABLED' else 'FAIL'} rule {rule_name}: "
              f"{rule.get('State')} -> {', '.join(t.get('Arn', '') for t in targets)}")

    # 4. Recent metrics: do the rules fire and is it delivered?
    cloudwatch = boto3.client("cloudwatch", region_name=sessions.region)
    report["metrics"] = {}
    for rule_name in (config.EVENT_RULE_FAIL, config.EVENT_RULE_START):
        triggered = _metric_sum(cloudwatch, "AWS/Events", "TriggeredRules",
                                [{"Name": "RuleName", "Value": rule_name}], hours)
        failed_invocations = _metric_sum(cloudwatch, "AWS/Events", "FailedInvocations",
                                         [{"Name": "RuleName", "Value": rule_name}], hours)
        report["metrics"][rule_name] = {"triggered": triggered,
                                        "failed_invocations": failed_invocations}
        print(f"Metrics {hours}h {rule_name} -> triggered: {triggered} | "
              f"failed invocations: {failed_invocations}")
    published = _metric_sum(cloudwatch, "AWS/SNS", "NumberOfMessagesPublished",
                            [{"Name": "TopicName", "Value": config.SNS_TOPIC_NAME}], hours)
    report["metrics"]["sns_messages_published"] = published
    print(f"Metrics {hours}h -> SNS published: {published}")

    ok = can_publish and report["has_confirmed_subscription"]
    print("=>", "alerts ready" if ok else "alerts BROKEN (see above)")
    return report
